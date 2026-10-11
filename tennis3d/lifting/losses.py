"""XYZ training losses in units of the fixed 10 m world scale."""

import torch


def position_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_valid: torch.Tensor,
    padding: torch.Tensor,
) -> torch.Tensor:
    """Average sum of three squared errors, including labelled occluded times.

    Both XYZ tensors are normalized by 10 m. Invalid labels may contain NaNs.
    """
    return (
        _position_errors(prediction, target, label_valid, padding)
        .square()
        .sum(-1)
        .mean()
    )


def radial_position_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_valid: torch.Tensor,
    padding: torch.Tensor,
) -> torch.Tensor:
    """Mean smooth Euclidean distance, including labelled occluded times.

    XYZ is divided by 10 m. Smooth at 0.1 m, subtract the zero-error offset,
    and scale by 0.1 to keep the radial gradient magnitude modest.
    """
    errors = _position_errors(prediction, target, label_valid, padding)
    return 0.1 * (torch.sqrt(errors.square().sum(-1) + 0.01**2) - 0.01).mean()


def vertical_radial_position_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_valid: torch.Tensor,
    padding: torch.Tensor,
) -> torch.Tensor:
    """Mean smooth radial error with vertical-coordinate gain 1.5.

    XYZ is divided by 10 m. Apply the gain before the norm, smooth the
    weighted radius at 0.1 m, subtract the zero-error offset and scale by 0.1.
    Include every valid labelled time, including occluded times.
    """
    errors = _position_errors(prediction, target, label_valid, padding)
    gains = errors.new_tensor((1.0, 1.0, 1.5))
    return 0.1 * (torch.sqrt((errors * gains).square().sum(-1) + 0.01**2) - 0.01).mean()


def _position_errors(prediction, target, label_valid, padding):
    if prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must have matching (...,3) shapes")
    if label_valid.shape != prediction.shape[:-1] or padding.shape != label_valid.shape:
        raise ValueError("loss masks must match XYZ time dimensions")
    valid = label_valid & ~padding
    if not valid.any():
        raise ValueError("position loss has no valid labels")
    pred, truth = prediction[valid], target[valid]
    if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
        raise ValueError("valid predictions and labels must be finite")
    return pred - truth
