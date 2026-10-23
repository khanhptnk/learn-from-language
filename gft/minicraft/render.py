"""Pixel rendering of MiniCraft states (for the visual-observation variant of the policy).

Each cell becomes a T x T RGB tile with a distinct colour and a simple pattern; the agent is drawn over its cell with a
dark notch on the side it faces. Rendering runs on the policy's device from the symbolic observation tensor (one-hot
cells + agent/facing channels, see learn.obs_tensor), so stored episodes stay compact and the network sees only pixels.
"""

import numpy as np
import torch

from gft.minicraft.core import (BRIDGE, EMPTY, FURNACE, GEM, N_CELLS, ORE, ROCK, TREE, WALL, WATER,
                                WORKBENCH)

TILE = 6


def _tiles(T=TILE):
    """(N_CELLS, T, T, 3) float tiles in [0, 1]."""
    def solid(rgb):
        return np.tile(np.array(rgb, float)[None, None], (T, T, 1)) / 255
    t = np.zeros((N_CELLS, T, T, 3))
    t[EMPTY] = solid((120, 190, 90))                       # grass
    t[WALL] = solid((70, 70, 70))
    t[WALL][::2, ::2] = np.array((50, 50, 50)) / 255      # stone-wall speckle
    t[TREE] = solid((120, 190, 90))
    t[TREE][1:T - 2, 1:T - 1] = np.array((30, 110, 40)) / 255    # canopy
    t[TREE][T - 2:, T // 2 - 1:T // 2 + 1] = np.array((110, 70, 30)) / 255   # trunk
    t[ROCK] = solid((150, 150, 150))
    t[ROCK][1::3, 1::3] = np.array((110, 110, 110)) / 255
    t[ORE] = solid((150, 150, 150))
    t[ORE][1::2, 1::3] = np.array((210, 120, 40)) / 255   # orange flecks
    t[WATER] = solid((50, 100, 220))
    t[WATER][::3] = np.array((90, 140, 240)) / 255        # ripples
    t[BRIDGE] = solid((150, 100, 50))
    t[BRIDGE][:, ::2] = np.array((120, 80, 40)) / 255     # planks
    t[WORKBENCH] = solid((170, 120, 60))
    t[WORKBENCH][1:3, 1:T - 1] = np.array((90, 60, 30)) / 255    # tool rack
    t[FURNACE] = solid((90, 40, 40))
    t[FURNACE][T // 2:, 1:T - 1] = np.array((240, 120, 30)) / 255  # fire
    t[GEM] = solid((120, 190, 90))
    for i in range(T):                                     # diamond
        w = min(i, T - 1 - i)
        t[GEM][i, T // 2 - w:T // 2 + w + 1] = np.array((60, 220, 230)) / 255
    return t


_TILE_CACHE = {}


def render(x, T=TILE):
    """(B, 3, S*T, S*T) images from symbolic observations x = (B, N_CELLS + 5, S, S) (learn.obs_tensor)."""
    key = (x.device, T)
    if key not in _TILE_CACHE:
        _TILE_CACHE[key] = torch.as_tensor(_tiles(T), dtype=torch.float32, device=x.device)
    tiles = _TILE_CACHE[key]
    B, _, S, _ = x.shape
    cells = x[:, :N_CELLS].argmax(1)                       # (B, S, S)
    img = tiles[cells]                                     # (B, S, S, T, T, 3)
    agent = x[:, N_CELLS]                                  # (B, S, S) one-hot agent position
    face = x[:, N_CELLS + 1:N_CELLS + 5]                   # (B, 4, S, S) facing at the agent cell
    body = torch.zeros(T, T, 3, device=x.device)
    body[1:T - 1, 1:T - 1] = torch.tensor((240, 210, 40), device=x.device) / 255     # agent: yellow square
    notch = torch.zeros(4, T, T, 1, device=x.device)       # dark marker on the facing side: up, right, down, left
    notch[0, 0:2, T // 2 - 1:T // 2 + 1] = 1
    notch[1, T // 2 - 1:T // 2 + 1, T - 2:T] = 1
    notch[2, T - 2:T, T // 2 - 1:T // 2 + 1] = 1
    notch[3, T // 2 - 1:T // 2 + 1, 0:2] = 1
    a = agent[..., None, None, None]                       # (B, S, S, 1, 1, 1)
    inner = torch.zeros(T, T, 1, device=x.device)
    inner[1:T - 1, 1:T - 1] = 1
    img = img * (1 - a * inner) + a * body
    mark = torch.einsum("bdij,dtuc->bijtuc", face, notch)  # (B, S, S, T, T, 1)
    img = img * (1 - mark) + mark * torch.tensor((20, 20, 20), device=x.device) / 255
    return img.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, S * T, S * T)


def save_png(img, path):
    """Save one (3, H, W) image (upscaled 4x) as PNG."""
    import matplotlib.pyplot as plt
    a = img.detach().cpu().permute(1, 2, 0).numpy()
    a = np.kron(a, np.ones((4, 4, 1)))
    plt.imsave(path, np.clip(a, 0, 1))
