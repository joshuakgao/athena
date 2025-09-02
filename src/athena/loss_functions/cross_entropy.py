"""Module for CrossEntropyLoss implementation."""

import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """A simple wrapper around PyTorch's CrossEntropyLoss.

    This loss function is commonly used for multi-class classification tasks.
    """

    def __init__(self, cfg):
        """Init function for CrossEntropyLoss."""
        super(CrossEntropyLoss, self).__init__()
        # Initialize the base CrossEntropyLoss from PyTorch
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """Forward function for CrossEntropyLoss.

        Args:
            outputs (torch.Tensor): The raw logits from the model.
                                    Shape: (batch_size, num_classes)
            targets (torch.Tensor): The ground truth labels.
                                    Shape: (batch_size,)

        Returns:
            torch.Tensor: The computed cross-entropy loss.
        """
        return self.loss_function(outputs, targets)
