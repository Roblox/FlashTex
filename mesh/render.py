import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from typing import Optional

from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    DirectionalLights,
    RasterizationSettings,
    MeshRasterizer,
    HardPhongShader,
    SoftPhongShader,
    TexturesUV,
)


def draw_meshes(meshes, cameras=None, lights=None, draw_flat=False, image_size=512, training=False, return_uvs=False, uv_masks=None, device='cuda'):
    if meshes is None:
        return (
            torch.ones((1, 3, image_size, image_size)),
            torch.zeros((1, 1, image_size, image_size)),
            torch.zeros((1, 1, image_size, image_size))
        )

    meshes = meshes.to(device)

    if cameras is None:
        B = len(meshes)
        elev = torch.tensor([0.0])
        azim = torch.tensor([0.0])
        R, T = look_at_view_transform(dist=2.0, elev=elev, azim=azim)
        R = R.expand(B, -1, -1)
        T = T.expand(B, -1)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    sigma = 1e-4
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=(np.log(1. / 1e-4 - 1.)*sigma) if training else 0.0,
        faces_per_pixel=50 if training else 1,
        perspective_correct=True,
        bin_size=0
    )

    # Project image
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(meshes_world=meshes)
    if draw_flat:
        images = hard_rgb_blend(meshes.sample_textures(fragments), fragments, BlendParams(background_color=(0.0, 0.0, 0.0)))
    else:
        lights = DirectionalLights(device=device, direction=[[-1.0, 0.5, 1.0]]) if lights is None else lights
        shader = HardPhongShader(cameras=cameras, lights=lights, blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)), device=device)
        images = shader(fragments, meshes)

    # Extract normal map
    normals = normal_shader(fragments, meshes)

    # Project provided UV mask to obtain corresponding mask in Image Space
    if uv_masks is not None:
        masked_uv_mesh = meshes.clone()
        num_views, _, _, C = meshes.textures.maps_padded().shape
        masked_uv_mesh.textures = TexturesUV(
                verts_uvs=meshes.textures.verts_uvs_padded(),
                faces_uvs=meshes.textures.faces_uvs_padded(),
                maps=uv_masks.repeat(num_views, 1, 1, C),
        )
        projected_uv_masks = (hard_rgb_blend(masked_uv_mesh.sample_textures(fragments), fragments, BlendParams(background_color=(0.0, 0.0, 0.0))) > 0).float()[..., 0:1]

    depth = fragments.zbuf
    image = images[:,:,:,0:3].permute(0,3,1,2)
    mask = images[:,:,:,3:4].permute(0,3,1,2) if uv_masks is None else projected_uv_masks.permute(0,3,1,2)
    depth = depth.permute(0,3,1,2)
    return dict(images=image, masks=mask, depths=depth, normals=normals, fragments=fragments)


def blur_depth(x):
    # return x
    x_valid = (x > 0.0).float()
    x_blur = TF.gaussian_blur(x + (torch.randn_like(x) * 0.01), kernel_size=13, sigma=5.0)
    return x_blur * x_valid


def normalize_depth_01(x):
    x_flat = x.view(x.size(0), -1)

    x_valid = x_flat[x_flat >= 0.0]
    if x_valid.size(0) == 0:
        print('invalid depth')
        return x
    
    x_valid_min = x_valid.min()
    x_valid_max = x_valid.max()

    x_valid = (x_valid - x_valid_min) / (x_valid_max - x_valid_min)
    x_valid = 1.0 - x_valid
    x_valid = 0.1 + (x_valid * 0.9)

    x_norm = torch.zeros_like(x_flat)
    x_norm[x_flat >= 0.0] = x_valid

    return x_norm.view(x.size(0), x.size(1), x.size(2), x.size(3))


def normal_shader(fragments, meshes):
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]
    ones = torch.ones_like(fragments.bary_coords)
    # pixel_normals = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_normals)
    pixel_normals = interpolate_face_attributes(fragments.pix_to_face, ones, faces_normals)  # <~~~ Use fragments.bary_coords in place of ones for smoothed normals
    normals = pixel_normals[:, :, :, 0]
    normals_magnitude = normals.norm(dim=-1, keepdim=True).repeat(1, 1, 1, 3)
    normals = (normals / normals_magnitude)
    normals[normals_magnitude == 0] = 0
    return normals
