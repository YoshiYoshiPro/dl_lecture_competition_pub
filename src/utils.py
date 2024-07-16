import random
from enum import Enum, auto
from time import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

class EventRepresentation:
    def __init__(self):
        pass

    def convert(self, events):
        raise NotImplementedError

class VoxelGrid(EventRepresentation):
    def __init__(self, input_size: tuple, normalize: bool):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros(
            (input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]
        self.normalize = normalize

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events['p'].device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = events['t']
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = events['x'].int()
            y0 = events['y'].int()
            t0 = t_norm.int()

            value = 2*events['p']-1
            for xlim in [x0, x0+1]:
                for ylim in [y0, y0+1]:
                    for tlim in [t0, t0+1]:
                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (
                            ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim-events['x']).abs()) * (
                            1 - (ylim-events['y']).abs()) * (1 - (tlim - t_norm).abs())
                        index = H * W * tlim.long() + W * ylim.long() + xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                voxel_grid = self.normalize_voxel_grid(voxel_grid)

        return voxel_grid

    @staticmethod
    def normalize_voxel_grid(voxel_grid: torch.Tensor) -> torch.Tensor:
        mask = torch.nonzero(voxel_grid, as_tuple=True)
        if mask[0].size()[0] > 0:
            mean = voxel_grid[mask].mean()
            std = voxel_grid[mask].std()
            if std > 0:
                voxel_grid[mask] = (voxel_grid[mask] - mean) / std
            else:
                voxel_grid[mask] = voxel_grid[mask] - mean
        return voxel_grid

class PolarityCount(EventRepresentation):
    def __init__(self, input_size: tuple):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros(
            (input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events['p'].device)
            voxel_grid = self.voxel_grid.clone()

            x0 = events['x'].int()
            y0 = events['y'].int()

            for xlim in [x0, x0+1]:
                for ylim in [y0, y0+1]:
                    mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0)
                    interp_weights = (1 - (xlim-events['x']).abs()) * (1 - (ylim-events['y']).abs())
                    index = H * W * events['p'].long() + W * ylim.long() + xlim.long()

                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        return voxel_grid

def flow_16bit_to_float(flow_16bit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)

    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
    return flow_map, valid2D

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor, valid_mask: torch.Tensor = None):
    epe = torch.norm(pred_flow - gt_flow, p=2, dim=1)
    if valid_mask is not None:
        valid_mask = valid_mask.squeeze(1) if valid_mask.dim() == 4 else valid_mask
        valid_mask = valid_mask.expand_as(epe)
        epe = epe[valid_mask]
    return epe.mean()

def smooth_loss(flow):
    dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
    dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
    return (dx.mean() + dy.mean()) / 2

def warp_images(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp images according to optical flow"""
    B, C, H, W = img.size()
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float().to(img.device)

    vgrid = grid + flow

    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = F.grid_sample(img, vgrid, align_corners=True)
    return output

def compute_multiscale_loss(pred_flows, gt_flow, gt_valid_mask):
    total_loss = 0
    weights = [0.32, 0.08, 0.02, 0.01]
    for i, pred_flow in enumerate(pred_flows):
        scale = 2 ** (3 - i)
        scaled_gt_flow = F.interpolate(gt_flow, scale_factor=1/scale, mode='bilinear', align_corners=False)
        scaled_gt_valid_mask = F.interpolate(gt_valid_mask.float(), scale_factor=1/scale, mode='nearest').bool()

        if pred_flow.shape != scaled_gt_flow.shape:
            pred_flow = F.interpolate(pred_flow, size=scaled_gt_flow.shape[2:], mode='bilinear', align_corners=False)

        loss = compute_epe_error(pred_flow, scaled_gt_flow, scaled_gt_valid_mask) * weights[i]
        total_loss += loss
    return total_loss

def total_loss(pred_flows, gt_flow, gt_valid_mask, smooth_weight=0.1):
    multiscale_loss = compute_multiscale_loss(pred_flows, gt_flow, gt_valid_mask)
    smoothness_loss = smooth_loss(pred_flows[-1])  # use full resolution flow for smoothness
    return multiscale_loss + smooth_weight * smoothness_loss
