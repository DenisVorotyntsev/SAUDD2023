import torch
from matplotlib import pyplot as plt


def make_prediction_dino(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    """
    Make depth predictions using the DinoModel.

    Args:
        model (torch.nn.Module): The DinoModel.
        img (torch.Tensor): The input image tensor.

    Returns:
        torch.Tensor: The predicted depth tensor.
    """
    disparity_image = model(img)
    ans = torch.nn.functional.interpolate(
        disparity_image,
        size=tuple(img.shape[2:]),
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)
    ans = torch.clip(ans, 1e-8, 1_000)
    return ans


def save_predictions_batch(
    img: torch.Tensor, predictions: torch.Tensor, target: torch.Tensor, save_path: str
) -> None:
    """
    Save predicted, target, and original images by concatenating them along dimension 1.

    Parameters:
    img (torch.Tensor): The original image tensor.
    predictions (torch.Tensor): The predicted image tensor.
    target (torch.Tensor): The target image tensor.
    save_path (str): The path where the output image will be saved.

    Returns:
    None
    """
    # Scaling images to improve visibility
    images = torch.cat(
        (min_max_scale(img), min_max_scale(target), min_max_scale(predictions)), dim=1
    )
    plt.imshow(images)
    plt.savefig(save_path)
    plt.close()


def min_max_scale(x: torch.Tensor) -> torch.Tensor:
    """
    Perform min-max scaling on a tensor to a range of 0 to 1.

    Parameters:
    x (torch.Tensor): The input tensor.

    Returns:
    torch.Tensor: The scaled tensor.
    """
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8)
