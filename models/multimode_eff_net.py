from models.multimode_resnet import LyftModelMultiMode
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class LyftModelMultiModeEffNet(LyftModelMultiMode):
    def __init__(self, agent_cfg: dict, criterion: torch.nn.Module, lr: float, eff_net_name='efficientnet-b3',
                 exp_factor=0.96, decay_steps_freq=1000, num_modes=3):
        super().__init__(agent_cfg, criterion, lr, exp_factor, decay_steps_freq, num_modes)
        self.save_hyperparameters()

        self.eff_net_name = eff_net_name
        self.criterion = criterion
        self.agent_cfg = agent_cfg
        self.lr = lr
        self.exp_factor = exp_factor
        self.decay_steps_freq = decay_steps_freq
        self.num_modes = num_modes

        # self.backbone = resnet50(pretrained=True, progress=True)
        self.backbone = EfficientNet.from_pretrained(eff_net_name)

        num_history_channels = (self.agent_cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone._conv_stem = nn.Conv2d(
            num_in_channels,
            self.backbone._conv_stem.out_channels,
            kernel_size=self.backbone._conv_stem.kernel_size,
            stride=self.backbone._conv_stem.stride,
            padding=self.backbone._conv_stem.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        # backbone_out_features = 2048
        backbone_out_features = self.backbone._conv_head.out_channels

        # X, Y coords for the future positions (output shape: Bx50x2)
        num_targets = 2 * self.agent_cfg["model_params"]["future_num_frames"]
        self.num_preds = num_targets * self.num_modes

        self.backbone._fc = nn.Linear(in_features=backbone_out_features, out_features=4096)
        self.head = nn.Linear(in_features=4096, out_features=self.num_preds + self.num_modes)
