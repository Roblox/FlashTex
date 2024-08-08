import random
from dataclasses import dataclass, field

import envlight
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *


@threestudio.register("pbr-material")
class PBRMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        material_activation: str = "sigmoid"
        environment_texture: str = "load/lights/mud_road_puresky_1k.hdr"
        environment_scale: float = 2.0
        min_metallic: float = 0.0
        max_metallic: float = 0.9
        min_roughness: float = 0.08
        max_roughness: float = 0.9
        min_albedo: float = 0.2
        max_albedo: float = 0.94
        use_bump: bool = True

    cfg: Config

    def configure(self) -> None:
        self.requires_normal = True
        self.requires_tangent = self.cfg.use_bump

        self.light = envlight.EnvLight(
            self.cfg.environment_texture, scale=self.cfg.environment_scale
        )

        FG_LUT = torch.from_numpy(
            np.fromfile("load/lights/bsdf_256_256.bin", dtype=np.float32).reshape(
                1, 256, 256, 2
            )
        )
        self.register_buffer("FG_LUT", FG_LUT)

    def forward(
        self,
        features: Float[Tensor, "*B Nf"],
        viewdirs: Float[Tensor, "*B 3"],
        shading_normal: Float[Tensor, "B ... 3"],
        tangent: Optional[Float[Tensor, "B ... 3"]] = None,
        env_light: Callable = None,
        mean_albedo_diffuse: bool = False,
        **kwargs,
    ) -> Float[Tensor, "*B 3"]:
        prefix_shape = features.shape[:-1]

        material: Float[Tensor, "*B Nf"] = get_activation(self.cfg.material_activation)(
            features
        )
        albedo = (
            material[..., :3] * (self.cfg.max_albedo - self.cfg.min_albedo)
            + self.cfg.min_albedo
        )
        
        metallic = (
            material[..., 3:4] * (self.cfg.max_metallic - self.cfg.min_metallic)
            + self.cfg.min_metallic
        )
        # metallic = torch.ones_like(metallic) * self.cfg.max_metallic
        roughness = (
            material[..., 4:5] * (self.cfg.max_roughness - self.cfg.min_roughness)
            + self.cfg.min_roughness
        )

        if self.cfg.use_bump:
            assert tangent is not None
            # perturb_normal is a delta to the initialization [0, 0, 1]
            perturb_normal = (material[..., 5:8] * 2 - 1) + torch.tensor(
                [0, 0, 1], dtype=material.dtype, device=material.device
            )
            perturb_normal = F.normalize(perturb_normal.clamp(-1, 1), dim=-1)

            # apply normal perturbation in tangent space
            bitangent = F.normalize(torch.cross(tangent, shading_normal), dim=-1)
            shading_normal = (
                tangent * perturb_normal[..., 0:1]
                - bitangent * perturb_normal[..., 1:2]
                + shading_normal * perturb_normal[..., 2:3]
            )
            shading_normal = F.normalize(shading_normal, dim=-1)

        v = -viewdirs
        n_dot_v = (shading_normal * v).sum(-1, keepdim=True)
        reflective = n_dot_v * shading_normal * 2 - v

        diffuse_albedo = (1 - metallic) * albedo

        fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1)
        fg = dr.texture(
            self.FG_LUT,
            fg_uv.reshape(1, -1, 1, 2).contiguous(),
            filter_mode="linear",
            boundary_mode="clamp",
        ).reshape(*prefix_shape, 2)
        F0 = (1 - metallic) * 0.04 + metallic * albedo
        specular_albedo = F0 * fg[:, 0:1] + fg[:, 1:2]
        
        if env_light is None:

            diffuse_light = self.light(shading_normal)
            specular_light = self.light(reflective, roughness)
        
        else:
            
            diffuse_light = env_light(shading_normal)
            specular_light = env_light(reflective, roughness)

        color = diffuse_albedo * diffuse_light + specular_albedo * specular_light
        color = color.clamp(0.0, 1.0)
        
        
        out = {
            "color": color,
            "diffuse": (diffuse_albedo * diffuse_light).clamp(0.0, 1.0),
            "specular": (specular_albedo * specular_light).clamp(0.0, 1.0),
            "albedo": albedo,
            "metallic": metallic,
            "roughness": roughness,
            "tangent": tangent if self.cfg.use_bump else None
        }
        
        if mean_albedo_diffuse:
            albedo_mean = albedo.mean(dim=list(range(len(albedo.shape)-1)), keepdim=True) # *B x 3
            metallic_mean = metallic.mean(dim=list(range(len(albedo.shape)-1)), keepdim=True) # *B x 3
            diffuse_albedo_mean = (1 - metallic_mean) * albedo_mean
            color_mean = diffuse_albedo_mean * diffuse_light + specular_albedo * specular_light
            color_mean = color_mean.clamp(0.0, 1.0)
            out.update({"color_mean_albedo": color_mean})
        

        return out

    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:
        material: Float[Tensor, "*N Nf"] = get_activation(self.cfg.material_activation)(
            features
        )
        albedo = material[..., :3]
        metallic = (
            material[..., 3:4] * (self.cfg.max_metallic - self.cfg.min_metallic)
            + self.cfg.min_metallic
        )
        roughness = (
            material[..., 4:5] * (self.cfg.max_roughness - self.cfg.min_roughness)
            + self.cfg.min_roughness
        )

        out = {
            "albedo": albedo,
            "metallic": metallic,
            "roughness": roughness,
        }

        if self.cfg.use_bump:
            perturb_normal = (material[..., 5:8] * 2 - 1) + torch.tensor(
                [0, 0, 1], dtype=material.dtype, device=material.device
            )
            perturb_normal = F.normalize(perturb_normal.clamp(-1, 1), dim=-1)
            perturb_normal = (perturb_normal + 1) / 2
            out.update(
                {
                    "bump": perturb_normal,
                }
            )

        return out
