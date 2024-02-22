from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class DDPM(LightningModule):
    def __init__(
        self
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)



if __name__ == '__main__':
    pass
        