from configs.general_config import trainer_args
import torch
from torch import nn
from torchvision.models.resnet import resnet18, resnet50
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as pl
from pytorch_lightning import EvalResult


# TODO: add train_epoch_end
# TODO: add val_epoch_end

class LyftModel(pl.LightningModule):
    def __init__(self, agent_cfg: dict, criterion: torch.nn.Module, lr: float, exp_factor=0.96, decay_steps_freq=1000):
        super().__init__()
        self.save_hyperparameters()

        self.criterion = criterion
        self.agent_cfg = agent_cfg
        self.lr = lr
        self.exp_factor = exp_factor
        self.decay_steps_freq = decay_steps_freq

        self.backbone = resnet50(pretrained=True, progress=True)

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

        self.backbone.fc = nn.Linear(in_features=backbone_out_features, out_features=num_targets)

    def forward(self, x):
        # x = self.backbone.conv1(x)
        # x = self.backbone.bn1(x)
        # x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x)
        #
        # x = self.backbone.layer1(x)
        # x = self.backbone.layer2(x)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)
        #
        # x = self.backbone.avgpool(x)
        # x = torch.flatten(x, 1)
        #
        # x = self.head(x)
        # x = self.logit(x)

        x = self.backbone.forward(x)

        return x

    def training_step(self, batch, batch_idx) -> dict:
        # TODO: add TrainResult
        x, y = batch['image'], batch['target_positions']
        y_hat = self(x).reshape(y.shape)
        target_availabilities = batch["target_availabilities"].unsqueeze(-1).type_as(y_hat)
        loss = self.criterion(y_hat, y)
        loss = loss * target_availabilities
        loss = loss.mean()

        result = pl.TrainResult()
        result.log('train_loss', loss)

        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx) -> EvalResult:
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
