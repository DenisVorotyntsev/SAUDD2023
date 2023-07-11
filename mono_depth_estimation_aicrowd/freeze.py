import numpy as np
from mono_depth_estimation_aicrowd.my_models import DinoModel


def freeze_dino_model(
    model: DinoModel, epoch: int, freeze_after_epoch: int = 10
) -> DinoModel:
    """
    Freezes the DINO model's backbone layers based on the epoch.

    Args:
        model (DinoModel): The DINO model.
        epoch (int): The current epoch.
        freeze_after_epoch (int, optional): The epoch after which to freeze the backbone layers. Defaults to 10.

    Returns:
        DinoModel: The updated DINO model.
    """
    # Unfreeze all parameters
    for param in model.dino_model.parameters():
        param.requires_grad = True

    if epoch <= freeze_after_epoch:
        print("Freezing DINO backbone")
        for param in model.dino_model.parameters():
            param.requires_grad = False
    else:
        print("Using unfrozen backbone")

    clc_frozen_params(model)
    return model


def clc_frozen_params(model: DinoModel) -> None:
    """
    Calculates and prints the number of total and frozen parameters in the model.

    Args:
        model (DinoModel): The DINO model.
    """
    model_parameters = filter(lambda p: True, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of total parameters: {params/1000000:.2f}M")

    model_parameters = filter(lambda p: p.requires_grad == False, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of frozen parameters: {params/1000000:.2f}M")
