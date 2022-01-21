import torch
from torch import nn
import wandb
from pathlib import Path


class MyModelCheckpoint:
    """Callback that save the model at the end of everyepoch.

        Args:
            filepath: the where to save the mode.
            monitor: the name of the value to track.

        """

    def __init__(self, filepath: str, monitor: str) -> None:
        self.filepath = filepath
        self.monitor = monitor

        # track best model
        self.best_metric: float = 0.0

        # create directory
        Path(self.filepath).mkdir(parents=True, exist_ok=True)

    def __call__(self, model: nn.Module, epoch: int, valid_metric) -> None:
        valid_metric_value: float = valid_metric[self.monitor].avg
        acc = "top5"
        loss = "losses"
        valid_metric_value: float = valid_metric[acc].avg
        wandb.log(
            {
                "accuracy": valid_metric[acc].avg,
                "validation loss": valid_metric[loss].avg,
            }
        )
        if valid_metric_value > self.best_metric:
            self.best_metric = valid_metric_value
            # store old metric and save new model
            filename_state_dict = Path(self.filepath) / f"model_{epoch}_state_dict.pt"
            torch.save(model.state_dict(), filename_state_dict)
            filename = Path(self.filepath) / f"model_{epoch}.pt"
            torch.save(model, filename)

