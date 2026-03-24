import torch

@torch.no_grad()
def forecast_rollout(model, x0: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Recursive multi-step forecast (autoregressive rollout).

    Args:
        model: takes (B,4,5) -> returns (B,4)
        x0: initial window, shape (B,4,5)
        steps: how many future points to forecast

    Returns:
        y: predictions, shape (B,steps,4)
    """
    model.eval()

    assert x0.ndim == 3 and x0.shape[1] == 4 and x0.shape[2] == 5, \
        f"Expected x0 shape (B,4,5), got {tuple(x0.shape)}"
    assert steps >= 1

    x = x0.clone()
    preds = []

    for _ in range(steps):
        y_next = model(x)              # (B,4)
        y_next_t = y_next.unsqueeze(-1)  # (B,4,1)

        # save in (B,1,4) format for easy stacking
        preds.append(y_next.unsqueeze(1))  # (B,1,4)

        # shift window left by 1 on time axis, append prediction at the end
        x = torch.cat([x[:, :, 1:], y_next_t], dim=2)  # (B,4,5)

    return torch.cat(preds, dim=1)  # (B,steps,4)
