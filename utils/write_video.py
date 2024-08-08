import os
import tempfile
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    DirectionalLights,
)

from mesh.render import draw_meshes


def write_360_video(mesh:Meshes, output_filename:str, walkaround_y:float, output_resolution:int=512):
    print('Creating 360 GIF...')
    
    device = 'cuda'

    temp_dir = tempfile.mkdtemp(prefix='depth2image_')

    num_frames = 48
    for frame_idx in range(num_frames):
        frame_t = float(frame_idx) / float(num_frames)
        camera_dist = 2.5
        orbit_angle = frame_t * np.pi * 2.0
        camera_x = camera_dist * np.sin(orbit_angle)
        camera_z = camera_dist * np.cos(orbit_angle)
        camera_y = np.sin(frame_t * np.pi * 2.0) * walkaround_y
        camera_pos = torch.tensor([[camera_x, camera_y, camera_z]]).float()
        camera_pos = camera_pos / (camera_pos**2).sum().sqrt() * camera_dist
        look_at_pos = torch.tensor([[0.0, 0.0, 0.0]])
        R, T = look_at_view_transform(eye=camera_pos, at=look_at_pos)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        light_dir = [camera_x, 0.5, camera_z]
        
        light_ambient = torch.tensor([[0.7, 0.7, 0.7]])
        light_diffuse = torch.tensor([[0.3, 0.3, 0.3]])
        light_specular = torch.tensor([[0.1, 0.1, 0.1]])        
        
        lights = DirectionalLights(device=device, direction=[light_dir], ambient_color=torch.clamp(light_ambient, 0.0, 1.0), diffuse_color=torch.clamp(light_diffuse, 0.0, 1.0), specular_color=torch.clamp(light_specular, 0.0, 1.0))

        generation_image_size = 1024
        rendered_data = draw_meshes(meshes=mesh, cameras=cameras, lights=lights, image_size=generation_image_size)
        mesh_images = rendered_data['images'] if output_resolution == generation_image_size else \
                      F.interpolate(rendered_data['images'], (output_resolution, output_resolution), mode='bilinear')
        frame_filename = os.path.join(temp_dir, f'walk_{str(frame_idx).zfill(4)}.png')
        torchvision.utils.save_image(mesh_images, frame_filename)

    # os.system(f"ffmpeg -y -r {num_frames} -i {temp_dir}/walk_%04d.png -c:v libx264 -vf fps={num_frames} -pix_fmt yuv420p {output_filename}")

    output_gif_filename = '.'.join(output_filename.split('.')[:-1]) + '.gif'
    os.system(f"convert -delay 30 -loop 0 {temp_dir}/walk_*.png {output_gif_filename}")
    
    shutil.rmtree(temp_dir)


def write_360_video_diffrast(renderer, output_filename, resolution=512, device='cuda', elev=0, camera_dist=5.0, fov=30, env_light=None):
    temp_dir = tempfile.mkdtemp(prefix='nvdiffrast')

    num_frames = 72
    for frame_idx in range(num_frames):
        frame_t = float(frame_idx) / float(num_frames)

        R, T = look_at_view_transform(dist=camera_dist, elev=elev, azim=frame_t * -360.0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)
        
        # c2w = cameras.get_world_to_view_transform().get_matrix().contiguous().to(device)  # Pytorch3d stores transposed matrix
        
        # print(cameras.get_world_to_view_transform().inverse().get_matrix().transpose(1,2).contiguous())
        c2w = cameras.get_world_to_view_transform().inverse().get_matrix().transpose(1,2).contiguous().to(device)
        
        camera_positions = cameras.get_camera_center().to(device)
        light_distances = 1.5
        light_direction = F.normalize(camera_positions, dim=-1)
        light_positions = light_distances * light_direction
        
        mvp_mtx = cameras.get_full_projection_transform().get_matrix().permute(0,2,1).contiguous()
        
        focal = fov_to_focal(fov, resolution)
        directions = get_ray_directions(resolution, resolution, focal).to(device)
        
        rays_o, rays_d = get_rays(directions, c2w)
        # print(c2w, rays_o[0], camera_positions[0])
        # print(((rays_o - camera_positions) ** 2).mean())
        

        with torch.no_grad():
            out = renderer(mvp_mtx=mvp_mtx, 
                            camera_positions=camera_positions,
                            light_positions=light_positions,
                            height=resolution, width=resolution,
                            c2w=c2w, rays_d=rays_d, env_bg=True, env_light=env_light,
                            mean_albedo_diffuse=False)


        # out['comp_normal_viewspace'] = 1 - out['comp_normal_viewspace']
            
        frame_filename = os.path.join(temp_dir, f'walk_rgb_{str(frame_idx).zfill(4)}.png')
        torchvision.utils.save_image(out['comp_rgb'].permute(0,3,1,2), frame_filename)
        frame_filename = os.path.join(temp_dir, f'walk_normal_{str(frame_idx).zfill(4)}.png')
        torchvision.utils.save_image(out['comp_normal_viewspace'].permute(0,3,1,2), frame_filename)
        
        # frame_filename = os.path.join(temp_dir, f'walk_normalR_{str(frame_idx).zfill(4)}.png')
        # torchvision.utils.save_image(out['comp_normal_viewspace'].permute(0,3,1,2)[:,0:1], frame_filename)
        # frame_filename = os.path.join(temp_dir, f'walk_normalG_{str(frame_idx).zfill(4)}.png')
        # torchvision.utils.save_image(out['comp_normal_viewspace'].permute(0,3,1,2)[:,1:2], frame_filename)
        # frame_filename = os.path.join(temp_dir, f'walk_normalB_{str(frame_idx).zfill(4)}.png')
        # torchvision.utils.save_image(out['comp_normal_viewspace'].permute(0,3,1,2)[:,2:3], frame_filename)
        
        if 'comp_rgb_diffuse' in out:
            frame_filename = os.path.join(temp_dir, f'walk_diffuse_{str(frame_idx).zfill(4)}.png')
            torchvision.utils.save_image(out['comp_rgb_diffuse'].permute(0,3,1,2), frame_filename)
            frame_filename = os.path.join(temp_dir, f'walk_specular_{str(frame_idx).zfill(4)}.png')
            torchvision.utils.save_image(out['comp_rgb_specular'].permute(0,3,1,2), frame_filename)

            # frame_filename = os.path.join(temp_dir, f'walk_mean_albedo_{str(frame_idx).zfill(4)}.png')
            # torchvision.utils.save_image(out['comp_rgb_mean_albedo'].permute(0,3,1,2), frame_filename)

            frame_filename = os.path.join(temp_dir, f'walk_albedo_{str(frame_idx).zfill(4)}.png')
            torchvision.utils.save_image(out['comp_albedo'].permute(0,3,1,2), frame_filename)
            frame_filename = os.path.join(temp_dir, f'walk_roughness_{str(frame_idx).zfill(4)}.png')
            torchvision.utils.save_image(out['comp_roughness'].permute(0,3,1,2), frame_filename)
            frame_filename = os.path.join(temp_dir, f'walk_metallic_{str(frame_idx).zfill(4)}.png')
            torchvision.utils.save_image(out['comp_metallic'].permute(0,3,1,2), frame_filename)
        
        if 'comp_tangent' in out:
            frame_filename = os.path.join(temp_dir, f'walk_tangent_{str(frame_idx).zfill(4)}.png')
            torchvision.utils.save_image(out['comp_tangent'].permute(0,3,1,2), frame_filename)
        

    # os.system(f"ffmpeg -y -r {num_frames} -i {temp_dir}/walk_%04d.png -c:v libx264 -vf fps={num_frames} -pix_fmt yuv420p {output_filename}")

    output_gif_filename = '.'.join(output_filename.split('.')[:-1]) + '_rgb.gif'
    os.system(f"convert -delay 10 -loop 0 {temp_dir}/walk_rgb_*.png {output_filename}")
    # output_gif_filename = '.'.join(output_filename.split('.')[:-1]) + '_normal.gif'
    # os.system(f"convert -delay 10 -loop 0 {temp_dir}/walk_normal_*.png {output_gif_filename}")
    # output_gif_filename = '.'.join(output_filename.split('.')[:-1]) + '_normalR.gif'
    # os.system(f"convert -delay 30 -loop 0 {temp_dir}/walk_normalR_*.png {output_gif_filename}")
    # output_gif_filename = '.'.join(output_filename.split('.')[:-1]) + '_normalG.gif'
    # os.system(f"convert -delay 30 -loop 0 {temp_dir}/walk_normalG_*.png {output_gif_filename}")
    # output_gif_filename = '.'.join(output_filename.split('.')[:-1]) + '_normalB.gif'
    # os.system(f"convert -delay 30 -loop 0 {temp_dir}/walk_normalB_*.png {output_gif_filename}")
    
    if 'comp_rgb_diffuse' in out:
        output_gif_filename = '.'.join(output_filename.split('.')[:-1]) + '_diffuse.gif'
        os.system(f"convert -delay 10 -loop 0 {temp_dir}/walk_diffuse_*.png {output_gif_filename}")
        output_gif_filename = '.'.join(output_filename.split('.')[:-1]) + '_specular.gif'
        os.system(f"convert -delay 10 -loop 0 {temp_dir}/walk_specular_*.png {output_gif_filename}")

        # output_gif_filename = '.'.join(output_filename.split('.')[:-1]) + '_mean_albedo.gif'
        # os.system(f"convert -delay 10 -loop 0 {temp_dir}/walk_mean_albedo_*.png {output_gif_filename}")

        output_gif_filename = '.'.join(output_filename.split('.')[:-1]) + '_albedo.gif'
        os.system(f"convert -delay 10 -loop 0 {temp_dir}/walk_albedo_*.png {output_gif_filename}")
        output_gif_filename = '.'.join(output_filename.split('.')[:-1]) + '_roughness.gif'
        os.system(f"convert -delay 10 -loop 0 {temp_dir}/walk_roughness_*.png {output_gif_filename}")
        output_gif_filename = '.'.join(output_filename.split('.')[:-1]) + '_metallic.gif'
        os.system(f"convert -delay 10 -loop 0 {temp_dir}/walk_metallic_*.png {output_gif_filename}")
    
    if 'comp_tangent' in out:
        output_gif_filename = '.'.join(output_filename.split('.')[:-1]) + '_tangent.gif'
        os.system(f"convert -delay 10 -loop 0 {temp_dir}/walk_tangent_*.png {output_gif_filename}")
    
    shutil.rmtree(temp_dir)

    
def fov_to_focal(fov, res):
    # fov : angle (-360, 360)
    fov_rad = fov / 180 * np.pi
    focal = res / (2 * np.tan(fov_rad / 2))
    return focal

def get_ray_directions(
    H,
    W,
    focal,
    use_pixel_centers = True,
):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    fx, fy = focal, focal
    cx, cy = W / 2, H / 2

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    )

    return directions


def get_rays(
    directions,
    c2w,
    keepdim=False,
    noise_scale=0.0,
):
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d
