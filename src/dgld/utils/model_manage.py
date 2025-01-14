"""Model Management"""

from pathlib import Path
from pathlib import PurePath
from typing import Tuple

import torch


def get_checkpoint_path(model_filename: str) -> PurePath:
    model_path: PurePath = Path(f"checkpoints/{model_filename}.pt")
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
    return model_path


def get_modelfile_path(model_filename: str) -> PurePath:
    model_path: PurePath = Path(f"checkpoints/{model_filename}.pt")
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
    return model_path


def check_modelfile_exists(model_filename: str) -> bool:
    if get_modelfile_path(model_filename).exists():
        return True
    return False


def save_model(
    model_filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    current_epoch: int,
    loss: float,
) -> None:
    """Save model, optimizer, current_epoch, loss to ``checkpoints/${model_filename}.pt``.

    Args:
        model_filename (str): filename to save model.
        model (torch.nn.Module): model.
        optimizer (torch.optim.Optimizer): optimizer.
        current_epoch (int): current epoch.
        loss (float): loss.
    """
    model_path = get_modelfile_path(model_filename)
    torch.save(
        {
            "epoch": current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        model_path,
    )


def load_model(
    model_filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """Load model from ``checkpoints/${model_filename}.pt``.

    Args:
        model_filename (str): filename to load model.
        model (torch.nn.Module): model.
        optimizer (torch.optim.Optimizer): optimizer.

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
            [model, optimizer, epoch, loss]
    """

    model_path = get_modelfile_path(model_filename)
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return model, optimizer, epoch, loss
