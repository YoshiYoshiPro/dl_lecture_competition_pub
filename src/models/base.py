import torch
from torch import nn
import torch.nn.functional as F

class build_resnet_block(nn.Module):
    """
    an improved resnet block which includes two general_conv2d
    """
    def __init__(self, channels, layers=2, do_batch_norm=False):
        super(build_resnet_block, self).__init__()
        self._channels = channels
        self._layers = layers

        self.res_block = nn.Sequential(*[general_conv2d(in_channels=self._channels,
                                             out_channels=self._channels,
                                             strides=1,
                                             do_batch_norm=do_batch_norm) for i in range(self._layers)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_res):
        identity = input_res
        out = self.res_block(input_res)
        out += identity
        return self.relu(out)

class upsample_conv2d_and_predict_flow(nn.Module):
    """
    an improved upsample convolution layer which includes a bilinear interpolate and a general_conv2d
    """
    def __init__(self, in_channels, out_channels, ksize=3, do_batch_norm=False):
        super(upsample_conv2d_and_predict_flow, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._ksize = ksize
        self._do_batch_norm = do_batch_norm

        self.general_conv2d = general_conv2d(in_channels=self._in_channels,
                                             out_channels=self._out_channels,
                                             ksize=self._ksize,
                                             strides=1,
                                             do_batch_norm=self._do_batch_norm,
                                             padding=1)  # Changed padding to 1
        
        self.predict_flow = nn.Conv2d(in_channels=self._out_channels,
                                      out_channels=2,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)

    def forward(self, conv):
        conv = F.interpolate(conv, scale_factor=2, mode='bilinear', align_corners=True)
        conv = self.general_conv2d(conv)

        flow = self.predict_flow(conv)
        flow = torch.tanh(flow) * 256.  # Apply tanh and scale
        
        return torch.cat([conv, flow], dim=1), flow

def general_conv2d(in_channels, out_channels, ksize=3, strides=2, padding=1, do_batch_norm=False, activation='leaky_relu'):
    """
    an improved general convolution layer which includes a conv2d, a leaky relu and a batch_normalize
    """
    layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize,
                        stride=strides, padding=padding)]
    
    if activation == 'leaky_relu':
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    elif activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    
    if do_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.99))
    
    return nn.Sequential(*layers)