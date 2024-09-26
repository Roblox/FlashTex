import random
from dataclasses import dataclass, field

from types import SimpleNamespace

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

class FourierEmbedding(nn.Module):
    def __init__(self, num_frequencies: int, include_input: bool = True, input_dim: int = 3):
        super(FourierEmbedding, self).__init__()

        self.input_dims = input_dim
        self.out_dims = 0
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        if include_input:
            self.out_dims += input_dim

        if num_frequencies >= 1:
            # 2 (sin, cos) * num_frequencies * input_dim
            frequency_dims = 2 * num_frequencies * input_dim
            self.out_dims += frequency_dims

        scales = 2.0 ** np.linspace(0.0, self.num_frequencies - 1, self.num_frequencies)
        self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32))

    def forward(self, x):
        assert x.shape[-1] == self.input_dims, f"Channel dimension is {x.shape[-1]} but should be {self.input_dims}"

        x_shape = list(x.shape)
        xb = (x[..., None] * self.scales).reshape(*x_shape[:-1], self.num_frequencies * x_shape[-1])

        four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))

        ret = [x] if self.include_input else []
        return torch.cat(ret + [four_feat], dim=-1)

    def get_output_dimensionality(self):
        return self.out_dims

class BrdfDecoder(nn.Module):
    def __init__(
        self,
        args,
        output_dimensions: int = 7,
        activation=nn.ReLU,
        final_activation=nn.Sigmoid,
        kernel_initializer=None,
        bias_initializer=None
    ):
        super(BrdfDecoder, self).__init__()

        self.latent_dim = args.latent_dim
        
        fourier_frequency_embedding = args.fourier_res

        embedder = FourierEmbedding(
            fourier_frequency_embedding, input_dim=args.latent_dim
        )
        print(args.latent_dim, embedder.get_output_dimensionality(), args.net_w)

        layers = []
        if fourier_frequency_embedding >= 1:
            layers.append(FourierEmbedding(fourier_frequency_embedding, input_dim=args.latent_dim))
            input_size = embedder.get_output_dimensionality()
        else:
            input_size = args.latent_dim

        layers.append(nn.Linear(input_size, args.net_w))
        layers.append(activation())

        for _ in range(args.net_d - 1):  # We've already added one layer above.
            layers.append(nn.Linear(args.net_w, args.net_w))
            layers.append(activation())
            if kernel_initializer:
                kernel_initializer(layers[-2].weight)
            if bias_initializer:
                bias_initializer(layers[-2].bias)

        print(args.net_w)

        final_layer = nn.Linear(args.net_w, output_dimensions)
        layers.append(final_layer)
        if final_activation:
            layers.append(final_activation())
        if kernel_initializer:
            kernel_initializer(final_layer.weight)
        if bias_initializer:
            bias_initializer(final_layer.bias)

        print(output_dimensions)

        self.net = nn.Sequential(*layers)
        print("Done")

    def forward(self, x):
        x = self.net(x)
        return x

    def random_sample(self, samples, mean=0.0, stddev=0.3):
        x = torch.clamp(
            torch.normal(mean=mean, std=stddev, size=(samples, self.latent_dim)).to(self.device),
            -1,
            1
        )
        return self(x)
    
    @property
    def device(self):
        return next(self.parameters()).device

@threestudio.register("latent-pbr-material")
class LatentPBRMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        # material_activation: str = "tanh"
        material_activation: str = "none"
        environment_texture: str = "load/lights/mud_road_puresky_1k.hdr"
        environment_scale: float = 2.0
        use_bump: bool = True
        latent_decoder_path: str = "/home/jovyan/output/brdf_ae/20231002-183400/checkpoint_1.pth"

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
        
        # Load the entire model's state dictionary
        checkpoint = torch.load(self.cfg.latent_decoder_path)

        # Extract only the decoder's state dictionary
        # This assumes a naming convention where your decoder parameters in the saved state dict start with "decoder."
        decoder_state_dict = {k[len("decoder."):]: v for k, v in checkpoint.items() if k.startswith("decoder.")}

        # Define a fresh decoder instance
        args = SimpleNamespace(net_d=5, net_w=64, latent_dim=4, fourier_res=8)  # Example arguments
        self.decoder = BrdfDecoder(args, output_dimensions=5)

        # Load the extracted state dict into the new decoder
        self.decoder.load_state_dict(decoder_state_dict)

        # Make sure to set the model to evaluation mode if you're doing inference
        self.decoder.eval()
    

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

        features_latent: Float[Tensor, "*B Nf"] = get_activation(self.cfg.material_activation)(
            features
        )
        
        material = self.decoder(features_latent)
        
        albedo = (
            material[..., :3]
        )
        
        metallic = (
            material[..., 4:5]
        )
        # metallic = torch.ones_like(metallic) * self.cfg.max_metallic
        roughness = (
            material[..., 3:4] 
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
            "roughness": roughness
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
        # material: Float[Tensor, "*N Nf"] = get_activation(self.cfg.material_activation)(
        #     features
        # )
        features_latent: Float[Tensor, "*B Nf"] = get_activation(self.cfg.material_activation)(
            features
        )
        
        material = self.decoder(features_latent)
        albedo = material[..., :3]
        
        metallic = (
            material[..., 4:5]
        )
        # metallic = torch.ones_like(metallic) * self.cfg.max_metallic
        roughness = (
            material[..., 3:4] 
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
