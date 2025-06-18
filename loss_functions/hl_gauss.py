import torch
import torch.nn as nn
import torch.nn.functional as F


class HLGaussLoss(nn.Module):
    """
    Calculates the KL Divergence between the model's output distribution
    and a Gaussian-smoothed target distribution.
    """

    def __init__(self, output_bins, sigma=1.0, device="cpu"):
        """
        Args:
            output_bins (int): The total number of output bins in the model.
            sigma (float): The standard deviation for the Gaussian curve.
                           A smaller sigma creates a sharper peak.
            device (str): The device to run the calculations on ('cpu' or 'cuda').
        """
        super(HLGaussLoss, self).__init__()
        self.output_bins = output_bins
        self.sigma = sigma
        self.device = device
        # Pre-compute a tensor of bin indices [0, 1, 2, ...]
        self.bins = torch.arange(self.output_bins, device=self.device).float()

    def forward(self, outputs, targets):
        """
        Args:
            outputs (torch.Tensor): The raw logits from the model.
                                    Shape: (batch_size, output_bins)
            targets (torch.Tensor): The one-hot encoded target tensor.
                                    Shape: (batch_size, output_bins)
        """
        # Get the index of the true bin for each sample in the batch
        true_bins = targets.argmax(dim=1).unsqueeze(1)  # Shape: (batch_size, 1)

        # Create the Gaussian-smoothed "soft" targets
        # Shape: (batch_size, output_bins)
        gauss_targets = torch.exp(-((self.bins - true_bins) ** 2) / (2 * self.sigma**2))
        # Normalize the distribution so it sums to 1
        gauss_targets = F.normalize(gauss_targets, p=1, dim=1)

        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(outputs, dim=1)

        # Calculate KL Divergence. 'batchmean' averages the loss over the batch.
        loss = F.kl_div(log_probs, gauss_targets, reduction="batchmean")

        return loss
