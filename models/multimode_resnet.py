from configs.general_config import trainer_args
import os

from utils import load_resnet
import torch
from torch import nn
from torchvision.models.resnet import resnet18, resnet50
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as pl
from pytorch_lightning import EvalResult


class LyftModelMultiMode(pl.LightningModule):
    def __init__(self, agent_cfg: dict, criterion: torch.nn.Module, lr: float,
                 exp_factor=0.96, decay_steps_freq=1000, num_modes=3):
        super().__init__()
        self.save_hyperparameters()

        self.criterion = criterion
        self.agent_cfg = agent_cfg
        self.lr = lr
        self.exp_factor = exp_factor
        self.decay_steps_freq = decay_steps_freq
        self.num_modes = num_modes

        self.backbone = load_resnet(resnet50)

        num_history_channels = (self.agent_cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        backbone_out_features = 2048

        # X, Y coords for the future positions (output shape: Bx50x2)
        num_targets = 2 * self.agent_cfg["model_params"]["future_num_frames"]
        self.num_preds = num_targets * self.num_modes

        self.backbone.fc = nn.Linear(in_features=backbone_out_features, out_features=4096)
        self.head = nn.Linear(in_features=4096, out_features=self.num_preds + self.num_modes)

    def forward(self, x):
        backbone_out = self.backbone.forward(x)
        head_out = self.head(backbone_out)

        bs, _ = head_out.shape
        pred, confidences = torch.split(head_out, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.agent_cfg["model_params"]["future_num_frames"], 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)

        return pred, confidences

    def training_step(self, batch, batch_idx) -> dict:
        x, y = batch['image'], batch['target_positions']
        pred, confidences = self.forward(x)
        target_availabilities = batch["target_availabilities"]
        loss = self.criterion(y, pred, confidences, target_availabilities)

        result = pl.TrainResult()
        result.log('loss', loss)

        return {'loss': loss, 'log': {'loss': loss}}

    def validation_step(self, batch, batch_idx) -> EvalResult:
        # TODO: add visualization of tracks
        loss_dict = self.training_step(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on=loss_dict['loss'])
        result.log('val_loss', loss_dict['loss'], prog_bar=True)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': ExponentialLR(optimizer, gamma=self.exp_factor),
            'interval': 'step',
            'frequency': self.decay_steps_freq
        }

        return [optimizer], [scheduler]

    def backward(self, trainer, loss, optimizer, optimizer_idx: int) -> None:
        loss.backward()
        for name, param in self.named_parameters():
            if not torch.isfinite(param.grad).all():
                print('Found error')
                print(param.grad)
