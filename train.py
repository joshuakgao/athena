import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from architecture import Athena
from datasets.aegis.dataset import AegisDataset
from utils.logger import logger

# Get training configs
with open("train_config.json", "r") as f:
    config = json.load(f)
    MODEL_NAME = config.get("model_name")
    NUM_EPOCHS = config["num_epochs"]
    LR = config["lr"]
    LR_DECAY_RATE = config.get("lr_decay_rate", 0.1)  # Default to 0.1
    BATCH_SIZE = config["batch_size"]
    TEST_SPLIT_RATIO = config.get("test_split_ratio", 0.2)  # Default to 20% test data
    USE_WANDB = config.get("use_wandb", False)
    NUM_RES_BLOCKS = config.get("num_res_blocks", 19)


# Create the dataset and split it into training and testing sets
dataset = AegisDataset()
dataset_size = len(dataset)
test_size = int(TEST_SPLIT_RATIO * dataset_size)
train_size = dataset_size - test_size
iters_in_an_epoch = max(len(dataset) // BATCH_SIZE, 1)
EVAL_MODEL_INTERVAL = max(iters_in_an_epoch // 10, 1)
CHECK_METRICS_INTERVAL = max(iters_in_an_epoch // 100, 1)
LR_DECAY_STEPS = iters_in_an_epoch


logger.info(
    f"Dataset size: {dataset_size}, Train size: {train_size}, Test size: {test_size}"
)

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Create the model, loss function, and optimizer
model = Athena(input_channels=9, num_res_blocks=NUM_RES_BLOCKS)
model.to(model.device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=LR_DECAY_STEPS, gamma=LR_DECAY_RATE
)

# Directory to save the models
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

if USE_WANDB:
    wandb.init(project="athena_chess", config=config, name=MODEL_NAME)
    wandb.watch(model)


def loss_function(pred, target):
    batch_size = pred.shape[0]

    from_indices = target[:, 0, :, :].view(batch_size, -1)
    to_indices = target[:, 1, :, :].view(batch_size, -1)

    # Extract "from" and "to" logits
    from_logits = pred[:, 0, :, :].view(batch_size, -1)  # [batch, 64]
    to_logits = pred[:, 1, :, :].view(batch_size, -1)  # [batch, 64]

    # Compute cross-entropy for "from" and "to" separately
    loss_from = F.cross_entropy(from_logits, from_indices)
    loss_to = F.cross_entropy(to_logits, to_indices)

    # Average the two losses
    loss = (loss_from + loss_to) / 2
    return loss


def eval_model(model, test_dataloader):
    model.eval()
    total_from_correct = 0
    total_to_correct = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for X, Y in test_dataloader:
            batch_size = X.size(0)  # Get actual batch size for this batch
            X = X.to(model.device)
            Y = Y.to(model.device)

            # Forward pass
            pred = model(X)

            # Compute loss
            loss = loss_function(pred, Y)
            total_loss += loss.item() * batch_size

            # Get predictions
            from_pred = pred[:, 0, :, :].view(batch_size, -1).argmax(dim=1)
            to_pred = pred[:, 1, :, :].view(batch_size, -1).argmax(dim=1)

            # Get targets
            from_target = Y[:, 0, :, :].view(batch_size, -1).argmax(dim=1)
            to_target = Y[:, 1, :, :].view(batch_size, -1).argmax(dim=1)

            # Update correct counts
            total_from_correct += (from_pred == from_target).sum().item()
            total_to_correct += (to_pred == to_target).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    from_accuracy = total_from_correct / total_samples
    to_accuracy = total_to_correct / total_samples
    overall_accuracy = (total_from_correct + total_to_correct) / (2 * total_samples)

    return {
        "loss": avg_loss,
        "from_accuracy": from_accuracy,
        "to_accuracy": to_accuracy,
        "overall_accuracy": overall_accuracy,
    }


# Training loop
logger.info("Training started")
best_accuracy = -1
for epoch in tqdm(range(NUM_EPOCHS)):
    # Learning rate decay
    if epoch > 0 and epoch % LR_DECAY_STEPS == 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= LR_DECAY_RATE
        logger.info(f"Reduced learning rate to {optimizer.param_groups[0]['lr']}")

    # Set model to training mode
    model.train()

    for batch_idx, (X, Y) in enumerate(train_dataloader):
        # Move data to device
        X = X.to(model.device)
        Y = Y.to(model.device)

        # Forward pass
        pred = model(X)

        # Compute loss
        loss = loss_function(pred, Y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log metrics
        if batch_idx % CHECK_METRICS_INTERVAL == CHECK_METRICS_INTERVAL - 1:
            logger.info(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss:.4f}"
            )
            if USE_WANDB:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )

        # Evaluate model periodically
        if batch_idx % EVAL_MODEL_INTERVAL == EVAL_MODEL_INTERVAL - 1:
            eval_metrics = eval_model(model, test_dataloader)
            logger.info(
                f"Evaluation - Loss: {eval_metrics['loss']:.4f}, "
                f"From Accuracy: {eval_metrics['from_accuracy']:.4f}, "
                f"To Accuracy: {eval_metrics['to_accuracy']:.4f}, "
                f"Overall Accuracy: {eval_metrics['overall_accuracy']:.4f}"
            )

            if USE_WANDB:
                wandb.log(
                    {
                        "eval_loss": eval_metrics["loss"],
                        "eval_from_accuracy": eval_metrics["from_accuracy"],
                        "eval_to_accuracy": eval_metrics["to_accuracy"],
                        "eval_overall_accuracy": eval_metrics["overall_accuracy"],
                    }
                )

            # Save best model
            if eval_metrics["overall_accuracy"] > best_accuracy:
                best_accuracy = eval_metrics["overall_accuracy"]
                torch.save(model.state_dict(), f"{save_dir}/best_model_{MODEL_NAME}.pt")
                logger.info(f"New best model saved with accuracy {best_accuracy:.4f}")

    # Log epoch-level metrics
    if USE_WANDB:
        wandb.log({"epoch": epoch + 1})

if USE_WANDB:
    wandb.finish()
