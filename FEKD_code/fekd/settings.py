"""训练设置无预设值，全部由命令行构造。"""

from dataclasses import dataclass
import math


@dataclass
class TrainingSettings:
    epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    nesterov: bool
    milestones: list[int]
    lr_decay: float
    max_grad_norm: float
    workers: int
    cpu_threads: int

    def validate(self):
        for key, minimum in (("epochs", 1), ("batch_size", 2), ("workers", 0), ("cpu_threads", 1)):
            if type(getattr(self, key)) is not int or getattr(self, key) < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        for key in ("learning_rate", "lr_decay"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("momentum", "weight_decay", "max_grad_norm"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.nesterov and self.momentum <= 0:
            raise ValueError("Nesterov requires positive momentum")
        if any(type(x) is not int or x < 1 for x in self.milestones) or self.milestones != sorted(
            set(self.milestones)
        ):
            raise ValueError("milestones must be sorted, unique positive integers")
