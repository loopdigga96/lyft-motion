from configs.general_config import trainer_args
from utils import load_resnet
import torch
from torch import nn
from torchvision.models.resnet import resnet18, resnet50
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as pl
from pytorch_lightning import EvalResult
import os


class LyftModelMultiModeAddFeatures(pl.LightningModule):
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

        history_size = self.agent_cfg["model_params"]["history_num_frames"] + 1
        num_history_channels = history_size * 2
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
        backbone_fc_out_features = 4096
        lstm_out = 10
        # concatenated backbone out + previous positions, previous yaws
        # aggregated_fc_features = backbone_fc_out_features + (history_size * 2) + history_size
        aggregated_fc_features = backbone_fc_out_features + (lstm_out * 2)

        # X, Y coords for the future positions (output shape: Bx50x2)
        num_targets = 2 * self.agent_cfg["model_params"]["future_num_frames"]
        self.num_preds = num_targets * self.num_modes

        self.backbone.fc = nn.Linear(in_features=backbone_out_features, out_features=backbone_fc_out_features)

        # self.hist_positions_fc = nn.Linear(in_features=history_size * 2, out_features=history_size * 2)
        # self.hist_yaws_fc = nn.Linear(in_features=history_size, out_features=history_size)
        self.hist_positions_lstm = nn.LSTM(input_size=2, hidden_size=lstm_out, batch_first=True)
        self.hist_yaws_lstm = nn.LSTM(input_size=1, hidden_size=lstm_out, batch_first=True)

        self.aggregated_fc = nn.Linear(in_features=aggregated_fc_features,
                                       out_features=aggregated_fc_features)
        self.head = nn.Linear(in_features=aggregated_fc_features, out_features=self.num_preds + self.num_modes)

    def forward(self, x, hist_positions, hist_yaws):
        backbone_out = self.backbone.forward(x)
        lstm_hist_pos_out, hidden_states = self.hist_positions_lstm.forward(hist_positions)
        lstm_hist_yaws_out, hidden_states = self.hist_yaws_lstm.forward(hist_yaws)
        # hist_positions = self.hist_positions_fc.forward(hist_positions)
        # hist_yaws = self.hist_yaws_fc.forward(hist_yaws)

        last_hist_pos_state = lstm_hist_pos_out[:, -1, :]
        last_hist_yaws_state = lstm_hist_yaws_out[:, -1, :]
        aggregated = torch.cat([backbone_out, last_hist_pos_state, last_hist_yaws_state], dim=1)
        aggregated = self.aggregated_fc.forward(aggregated)
        head_out = self.head(aggregated)

        bs, _ = head_out.shape
        pred, confidences = torch.split(head_out, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.agent_cfg["model_params"]["future_num_frames"], 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)

        return pred, confidences

    def training_step(self, batch, batch_idx) -> dict:
        x, y, hist_positions, history_yaws = (batch['image'], batch['target_positions'],
                                              batch['history_positions'], batch['history_yaws'])
        # batch_size = x.size(0)
        # hist_positions = hist_positions.view(batch_size, -1)
        # history_yaws = history_yaws.view(batch_size, -1)
        pred, confidences = self.forward(x, hist_positions, history_yaws)
        target_availabilities = batch["target_availabilities"]
        loss = self.criterion(y, pred, confidences, target_availabilities)

        result = pl.TrainResult()
        result.log('loss', loss)

        if batch_idx == 0:
            sample_input = (torch.rand_like(x),
                            torch.rand_like(hist_positions),
                            torch.rand_like(history_yaws))
            self.logger.experiment.add_graph(self, sample_input)

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
        # for name, param in self.named_parameters():
        #     if not torch.isfinite(param.grad).all():
        #         print('Found error')
        #         print(param.grad)

