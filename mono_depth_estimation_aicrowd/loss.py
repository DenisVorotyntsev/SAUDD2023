import torch
from torch import nn
from torch import Tensor


def si_log(target: Tensor, prediction: Tensor) -> Tensor:
    """
    Calculates the Scale-Invariant Logarithmic Error (SI Log) score between the target and prediction tensors.

    Args:
        target (Tensor): The target tensor.
        prediction (Tensor): The prediction tensor.

    Returns:
        Tensor: The SI Log score.
    """
    mask = target > 0
    num_vals = mask.sum()
    log_diff = torch.log(prediction[mask]) - torch.log(target[mask])
    si_log_unscaled = torch.sum(log_diff**2) / num_vals - (
        torch.sum(log_diff) ** 2
    ) / (num_vals**2)
    si_log_score = torch.sqrt(si_log_unscaled) * 100
    return si_log_score


class SILoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
        Calculates the Scale-Invariant Logarithmic Loss (SI Loss) between the prediction and target tensors.

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.

        Returns:
            Tensor: The SI Loss.
        """
        bs = prediction.shape[0]

        prediction = torch.reshape(prediction, (bs, -1))
        target = torch.reshape(target, (bs, -1))

        mask = target > 0
        num_vals = mask.sum(dim=1)

        log_diff = torch.zeros_like(prediction)
        log_diff[mask] = torch.log(prediction[mask]) - torch.log(target[mask])

        si_log_unscaled = torch.sum(log_diff**2, dim=1) / num_vals - (
            torch.sum(log_diff, dim=1) ** 2
        ) / (num_vals**2)
        si_log_score = torch.sqrt(si_log_unscaled) * 100

        si_log_score = torch.mean(si_log_score)
        return si_log_score
