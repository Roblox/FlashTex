from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *

import envlight


@threestudio.register("envlight-background")
class EnvlightBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        environment_texture: str = "load/lights/mud_road_puresky_1k.hdr"
        # environment_texture: str = "load/lights/aerodynamics_workshop_2k.hdr"
        environment_scale: float = 1.0

    cfg: Config

    def configure(self) -> None:
        self.light = envlight.EnvLight(
            self.cfg.environment_texture, scale=self.cfg.environment_scale
        )

    def forward(self, dirs: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B Nc"]:
        dirs_shape = dirs.shape[:-1]
        color = self.light(dirs, torch.zeros_like(dirs[...,0:1]))
        # color = dirs
        return color
