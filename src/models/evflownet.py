import torch
import torch.nn.functional as F
from torch import nn
from src.models.base import *
from typing import List, Dict, Any

class EVFlowNet(nn.Module):
    def __init__(self, args):
        super(EVFlowNet, self).__init__()
        self._args = args

        self.encoder1 = self._make_encoder_layer(4, 64)
        self.encoder2 = self._make_encoder_layer(64, 128)
        self.encoder3 = self._make_encoder_layer(128, 256)
        self.encoder4 = self._make_encoder_layer(256, 512)

        self.decoder1 = self._make_decoder_layer(512, 256)
        self.decoder2 = self._make_decoder_layer(512, 128)
        self.decoder3 = self._make_decoder_layer(256, 64)
        self.decoder4 = self._make_decoder_layer(128, 32)

        self.flow_pred1 = nn.Conv2d(256, 2, kernel_size=3, padding=1)
        self.flow_pred2 = nn.Conv2d(128, 2, kernel_size=3, padding=1)
        self.flow_pred3 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.flow_pred4 = nn.Conv2d(32, 2, kernel_size=3, padding=1)

    def _make_encoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            general_conv2d(in_channels, out_channels, do_batch_norm=not self._args.no_batch_norm),
            nn.Dropout(0.2)
        )

    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels) if not self._args.no_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # inputs is a list of tensors at different scales
        x = inputs[0]  # full resolution input

        # Encoder
        skip_connections = []
        for i, encoder in enumerate([self.encoder1, self.encoder2, self.encoder3, self.encoder4]):
            x = encoder(x)
            if i < 3:  # don't save skip connection for last encoder layer
                skip_connections.append(x)
            if i < len(inputs) - 1:
                resized_input = F.interpolate(inputs[i+1], size=x.shape[2:], mode='bilinear', align_corners=False)
                if resized_input.shape[1] != x.shape[1]:
                    resized_input = self.adjust_channels(resized_input, x.shape[1])
                x = x + resized_input

        # Decoder
        flow_outputs = []
        x = self.decoder1(x)
        flow1 = self.flow_pred1(x)
        flow_outputs.append(flow1)

        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.decoder2(x)
        flow2 = self.flow_pred2(x)
        flow_outputs.append(flow2)

        x = torch.cat([x, skip_connections[-2]], dim=1)
        x = self.decoder3(x)
        flow3 = self.flow_pred3(x)
        flow_outputs.append(flow3)

        x = torch.cat([x, skip_connections[-3]], dim=1)
        x = self.decoder4(x)
        flow4 = self.flow_pred4(x)
        flow_outputs.append(flow4)

        return flow_outputs

    def adjust_channels(self, x: torch.Tensor, target_channels: int) -> torch.Tensor:
        current_channels = x.shape[1]
        if current_channels < target_channels:
            return F.pad(x, (0, 0, 0, 0, 0, target_channels - current_channels))
        elif current_channels > target_channels:
            return x[:, :target_channels, :, :]
        else:
            return x