import numpy as np


class EarlyStopper:
    """
    Class for implementing early stopping based on validation loss.
    """

    def __init__(self, patience: int = 1, min_delta: float = 0):
        """
        Initialize the EarlyStopper.

        Args:
            patience (int): Number of epochs to wait before stopping if the validation loss does not improve.
            min_delta (float): Minimum change in the validation loss to be considered as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float) -> bool:
        """
        Check if early stopping criteria are met based on the validation loss.

        Args:
            validation_loss (float): The validation loss value to compare.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
