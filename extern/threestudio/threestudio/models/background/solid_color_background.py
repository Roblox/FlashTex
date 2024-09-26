import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.utils.typing import *

import numpy as np
from threestudio.utils.color import *


@threestudio.register("solid-color-background")
class SolidColorBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        color: Tuple = (1.0, 1.0, 1.0)
        learned: bool = False
        random_aug: bool = False
        random_aug_prob: float = 0.5
        hls_color: bool = True
        h_range: Tuple = (0.0, 1.0)
        l_range: Tuple = (0.0, 1.0)
        s_range: Tuple = (0.0, 1.0)

    cfg: Config

    def configure(self) -> None:
        self.env_color: Float[Tensor, "Nc"]
        if self.cfg.learned:
            self.env_color = nn.Parameter(
                torch.as_tensor(self.cfg.color, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "env_color", torch.as_tensor(self.cfg.color, dtype=torch.float32)
            )

    def forward(self, dirs: Float[Tensor, "B H W 3"]) -> Float[Tensor, "B H W Nc"]:
        color = (
            torch.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(dirs)
            * self.env_color
        )
        if (
            self.training
            and self.cfg.random_aug
            and random.random() < self.cfg.random_aug_prob
        ):
            if self.cfg.hls_color:
                N = dirs.shape[0]
                h_rand = np.random.rand(N) * (self.cfg.h_range[1] - self.cfg.h_range[0]) + self.cfg.h_range[0]
                l_rand = np.random.rand(N) * (self.cfg.l_range[1] - self.cfg.l_range[0]) + self.cfg.l_range[0]
                s_rand = np.random.rand(N) * (self.cfg.s_range[1] - self.cfg.s_range[0]) + self.cfg.s_range[0]
                hls_rand = np.stack([h_rand, l_rand, s_rand], axis=-1)
                rgb_rand = hls_to_rgb(hls_rand)
                rgb_rand = torch.from_numpy(rgb_rand).reshape(N, 1, 1, 3).to(dirs).expand(*dirs.shape[:-1], -1)
                color = color * 0 + rgb_rand
                
            else:
                # use random background color with probability random_aug_prob
                color = color * 0 + (  # prevent checking for unused parameters in DDP
                    torch.rand(dirs.shape[0], 1, 1, self.cfg.n_output_dims)
                    .to(dirs)
                    .expand(*dirs.shape[:-1], -1)
                )
        return color
