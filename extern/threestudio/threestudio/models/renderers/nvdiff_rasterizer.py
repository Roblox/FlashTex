from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("nvdiff-rasterizer")
class NVDiffRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_rgb: bool = True,
        env_bg: bool = False,
        env_light: Callable = None,
        perturb: bool = False,
        mean_albedo_diffuse: bool = False,
        rotate_envlight: Float[Tensor, "B 3"] = None, # euler angle: x,y,z
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]
        mesh = self.geometry.isosurface()

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out = {"opacity": mask_aa, "mesh": mesh}

        gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
        gb_normal = F.normalize(gb_normal, dim=-1)
        gb_normal_aa = torch.lerp(
            torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
        )
        gb_normal_aa = self.ctx.antialias(
            gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

        # Compute normal in view space (BAE).
        # TODO: make is clear whether to compute this.
        w2c = kwargs["c2w"][:, :3, :3].inverse()
        gb_normal_viewspace = torch.einsum("bij,bhwj->bhwi", w2c, gb_normal)
        gb_normal_viewspace = F.normalize(gb_normal_viewspace, dim=-1)
        bg_normal = torch.zeros_like(gb_normal_viewspace)
        # bg_normal[..., 2] = 1
        gb_normal_viewspace_aa = torch.lerp(
            # (bg_normal + 1.0) / 2.0,
            bg_normal,
            1 - (gb_normal_viewspace + 1.0) / 2.0,
            mask.float(),
        ).contiguous()
        gb_normal_viewspace_aa = self.ctx.antialias(
            gb_normal_viewspace_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"comp_normal_viewspace": gb_normal_viewspace_aa})

        # TODO: make it clear whether to compute the normal, now we compute it in all cases
        # consider using: require_normal_computation = render_normal or (render_rgb and material.requires_normal)
        # or
        # render_normal = render_normal or (render_rgb and material.requires_normal)

        if render_rgb:
            selector = mask[..., 0]

            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            # print(f"gb_pos shape, {gb_pos.shape}, rays_d shape, {kwargs['rays_d'].shape}")

            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )
            gb_light_positions = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )

            positions = gb_pos[selector]
            geo_out = self.geometry(positions, output_normal=False)
            
            if perturb:
                positions_perturb = positions + torch.normal(mean=0, std=0.01, size=positions.shape, device=positions.device)
                geo_out_perturb = self.geometry(positions_perturb, output_normal=False)

            extra_geo_info = {}
            if self.material.requires_normal:
                extra_geo_info["shading_normal"] = gb_normal[selector]
            if self.material.requires_tangent:
                gb_tangent, _ = self.ctx.interpolate_one(
                    mesh.v_tng, rast, mesh.t_pos_idx
                )
                gb_tangent = F.normalize(gb_tangent, dim=-1)
                extra_geo_info["tangent"] = gb_tangent[selector]

            out_material = self.material(
                viewdirs=gb_viewdirs[selector],
                positions=positions,
                light_positions=gb_light_positions[selector],
                env_light=env_light,
                mean_albedo_diffuse=mean_albedo_diffuse,
                **extra_geo_info,
                **geo_out
            )
            if isinstance(out_material, dict):
                rgb_fg = out_material['color']
                diffuse, specular = out_material['diffuse'], out_material['specular']
                
                gb_diffuse = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
                gb_diffuse[selector] = diffuse
                
                gb_specular = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
                gb_specular[selector] = specular
                
                out.update({"comp_rgb_diffuse": gb_diffuse, "comp_rgb_specular": gb_specular})
                
                albedo, roughness, metallic = out_material['albedo'], out_material['roughness'], out_material['metallic']
                gb_albedo = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
                gb_albedo[selector] = albedo
                gb_albedo = self.ctx.antialias(gb_albedo, rast, v_pos_clip, mesh.t_pos_idx)
                
                gb_roughness = torch.zeros(batch_size, height, width, 1).to(rgb_fg)
                gb_roughness[selector] = roughness
                gb_roughness = self.ctx.antialias(gb_roughness, rast, v_pos_clip, mesh.t_pos_idx)
                
                gb_metallic = torch.zeros(batch_size, height, width, 1).to(rgb_fg)
                gb_metallic[selector] = metallic
                gb_metallic = self.ctx.antialias(gb_metallic, rast, v_pos_clip, mesh.t_pos_idx)
                
                out.update({"comp_albedo": gb_albedo, "comp_roughness": gb_roughness, "comp_metallic": gb_metallic})
                
                if out_material['tangent'] is not None:
                    tangent = out_material['tangent']
                    gb_tangent = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
                    gb_tangent[selector] = tangent
                    gb_tangent = self.ctx.antialias(gb_tangent, rast, v_pos_clip, mesh.t_pos_idx)
                    
                    out.update({"comp_tangent": gb_tangent})
                
                
                if perturb:
                    out_material_perturb = self.material(
                        viewdirs=gb_viewdirs[selector],
                        positions=positions_perturb,
                        light_positions=gb_light_positions[selector],
                        env_light=env_light,
                        **extra_geo_info,
                        **geo_out_perturb
                    )
                    albedo_grad = torch.mean(torch.sum(torch.abs(out_material['albedo'] - out_material_perturb['albedo']), dim=-1, keepdim=True) / 3)
                    roughness_grad = torch.mean(torch.sum(torch.abs(out_material['roughness'] - out_material_perturb['roughness']), dim=-1, keepdim=True))
                    metallic_grad = torch.mean(torch.sum(torch.abs(out_material['metallic'] - out_material_perturb['metallic']), dim=-1, keepdim=True))
                    
                    out.update({"albedo_grad": albedo_grad, 'roughness_grad': roughness_grad, 'metallic_grad': metallic_grad})
                    
                    if out_material['tangent'] is not None:
                        tangent_grad = torch.mean(torch.sum(torch.abs(out_material['tangent'] - out_material_perturb['tangent']), dim=-1, keepdim=True))
                        out.update({"tangent_grad": tangent_grad})
                        
                
            else:
                rgb_fg = out_material
            gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
            gb_rgb_fg[selector] = rgb_fg

            if env_bg:
                assert 'rays_d' in kwargs
                # print(f"gb_pos shape, {gb_pos.shape}, rays_d shape, {kwargs['rays_d'].shape}")
                gb_bg_viewdirs = F.normalize(kwargs['rays_d'].reshape(-1, height, width, 3), dim=-1)
                if env_light is None:
                    gb_rgb_bg = self.background(dirs=gb_bg_viewdirs)
                else:
                    gb_rgb_bg = env_light(gb_bg_viewdirs, torch.zeros_like(gb_bg_viewdirs[...,0:1]))
            else:
                gb_rgb_bg = self.background(dirs=gb_viewdirs)
            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

            out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg})
            
            if mean_albedo_diffuse:
                rgb_mean_albedo = out_material['color_mean_albedo']
                gb_rgb_mean_albedo = torch.zeros(batch_size, height, width, 3).to(rgb_mean_albedo)
                gb_rgb_mean_albedo[selector] = rgb_mean_albedo
                gb_rgb_mean_albedo = torch.lerp(gb_rgb_bg, gb_rgb_mean_albedo, mask.float())
                gb_rgb_mean_albedo_aa = self.ctx.antialias(gb_rgb_mean_albedo, rast, v_pos_clip, mesh.t_pos_idx)
                
                out.update({"comp_rgb_mean_albedo": gb_rgb_mean_albedo_aa})
                

        return out
