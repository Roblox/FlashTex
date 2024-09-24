import os
import time
import sys
project_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, project_path)

threestudio_path = os.path.join(project_path, "extern/threestudio")
sys.path.insert(0, threestudio_path)

from argparse import ArgumentParser
import pathlib
import tempfile
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import json
from concurrent.futures import ThreadPoolExecutor

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    TexturesUV,
)

from mesh.util import (
    load_mesh,
    write_obj_with_texture
)
from dataset.mesh_dataset import MeshDataset
from models.avatar_image_generator import (
    AvatarImageGenerator,
    to_character_sheet,
    from_character_sheet
)

from optimization.optimizer3d import Optimizer3D
from optimization.setup_geometry3d import setup_geometry3d
from utils.write_video import (
    write_360_video,
    write_360_video_diffrast
)


import threestudio

from threestudio.models.renderers.nvdiff_rasterizer import NVDiffRasterizer
from threestudio.models.materials.no_material import NoMaterial
from threestudio.models.materials.pbr_material import PBRMaterial
from threestudio.models.background.solid_color_background import SolidColorBackground


def parse_args(arglist=None):
    expand_path = lambda p: pathlib.Path(p).expanduser().absolute() if p is not None else None
    parser = ArgumentParser(description='FlashTex')
    parser.add_argument('--input_mesh', 
                        type=expand_path, 
                        help='Path to input mesh')
    parser.add_argument('--output', dest='output_dir',
                        type=expand_path,
                        default='./output', help='Path to output directory')
    parser.add_argument('--production', action='store_true', help='Run in production mode, skipping debug outputs')
    parser.add_argument('--model_id', type=str, default='Lykon/DreamShaper', help='Diffusers model to use for generation')
    parser.add_argument('--controlnet_name', type=str, default='', help='ControlNet model to use for generation')
    parser.add_argument('--pretrained_dir', type=str, help='Directory containing pretrained weights for models')
    parser.add_argument('--distilled_encoder', type=str, default='load/encoder_resnet4.pth', help='Disilled encoder checkpoint')
    parser.add_argument('--image_resolution', type=int, default=512, help='Image resolution')
    parser.add_argument('--num_sds_iterations', type=int, default=400, help='Number of iterations for SDS optimization')
    parser.add_argument('--rotation_x', type=float, default=0.0, help='Mesh rotation about the X axis')
    parser.add_argument('--rotation_y', type=float, default=0.0, help='Mesh rotation about the Y axis')
    parser.add_argument('--gif_resolution', type=int, default=512, help='Resolution of spin-around gif')
    parser.add_argument('--refine', action='store_true', help='Refine original mesh texture')
    parser.add_argument('--bbox_size', type=float, default=-1, help='Size of a mesh bbox enclosing mesh avatar/object')
    parser.add_argument('--texture_tile_size', type=int, default=1024, help='Size each texture tile in UV space')
    parser.add_argument('--uv_unwrap', action='store_true', help='Perform uv unwrapping')
    parser.add_argument('--uv_rescale', action='store_true', help='Perform uv rescaling')
    
    # Arguments for generating the reference image
    parser.add_argument('--disable_img2img', action='store_true', help='Do not use img2img for the character sheet')
    parser.add_argument('--ddim_steps', type=int, default=20, help='DDIM steps')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--img2img_strength', type=float, default=1.0, help='Strength for img2img for the character sheet')
    parser.add_argument('--character_sheet_noise', type=float, default=0.0, help='Character sheet noise scale')
    parser.add_argument('--strength', type=float, default=0.8, help='Strength')
    parser.add_argument('--scale', type=float, default=9.0, help='Scale')
    parser.add_argument('--eta', type=float, default=0.0, help='Eta')

    parser.add_argument('--camera_dist', type=float, default=5.0, help='Camera distance')
    parser.add_argument('--camera_fov', type=float, default=30.0, help='Camera FOV')
    parser.add_argument('--walkaround_y', type=float, default=0.0, help='Walkaround camera Y')
    parser.add_argument('--skip_character_sheet', action='store_true', help='Set to skip character sheet for multiview consistency')
    parser.add_argument('--prompt_masking', dest='prompt_masking_style', type=str, default='global', help='global | front_back | front_back_localized')
    parser.add_argument('--prompt', type=str, default='a mouse pirate, detailed, hd', help='Text prompt for stable diffusion')
    parser.add_argument('--additional_prompt', dest="a_prompt", type=str, default="", help='Additional text prompt for stable diffusion')
    parser.add_argument('--negative_prompt', dest="n_prompt", type=str, default="nude, naked, bad quality, blurred, low resolution, low quality, low res", help='Negative text prompt for stable diffusion')  
    parser.add_argument('--device', type=str, default="cuda", help='Device to use (cpu or cuda), defaults to cuda when available.')
    parser.add_argument('--guidance_scale', type=float, default=50.0, help='Guidance Scale')
    parser.add_argument('--cond_strength', type=float, default=1.0, help='Condtioning Strength for ControlNet')
    parser.add_argument('--guidance_sds', type=str, default="SDS_sd", help='Choose from [SDS_sd, SDS_LightControlNet]')
    parser.add_argument('--no_tqdm', action='store_true', help='No tqdm logging')
    parser.add_argument('--SDS_camera_dist', type=float, default=5.0)
    parser.add_argument('--pbr_material', action='store_true', help='Use PBR Material.')
    parser.add_argument('--lambda_recon_reg', type=float, default=1000.0, help='Reconstruction regularization')
    parser.add_argument('--lambda_albedo_smooth', type=float, default=5.0, help='Albedo smoothness regularization')
    args = parser.parse_args(args=arglist)
    return args

def view_angle_to_prompt(elev, azim):
    azim = azim % 360
    if abs(azim - 180.0) < 90.0:
        return 'rear view'
    elif abs(azim) < 30.0 or abs(azim - 360) < 30:
        return 'front view'
    else:
        return 'side view'
    

def args_to_json(args):
    args = vars(args)
    for key in args.keys():
        value = args[key]
        if isinstance(value, pathlib.Path):
            value = str(value.absolute())
        args[key] = value
    return json.dumps(args, indent=2)


def get_view_params(mode='character_sheet', num_views=2):
    if mode == 'character_sheet':
        if num_views == 2:
            elev = torch.tensor([0.0, 0.0])
            azim = torch.tensor([0.0, 180.0])
            light_dirs = [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        elif num_views == 4:
            elev = torch.tensor([0.0, 0.0, 15.0, 15.0])
            azim = torch.tensor([0.0, 180.0, -75.0, 75.0])
            light_dirs = [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        else:
            raise NotImplementedError(f'Unsupported number of views {num_views}')
    else:
        raise NotImplementedError(f'Unsupported view mode {mode}')
        
    return elev, azim, light_dirs


def setup(args):        
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.production:
        with open(f'{args.output_dir}/config.json', 'w') as file:
            file.write(args_to_json(args))

    return args

        
def get_mesh_dict_and_view_strings(args, use_textures=False):
    mesh_dataset = MeshDataset(input_mesh=args.input_mesh,
                            device=args.device,
                            head_only=False,
                            texture_tile_size=args.texture_tile_size,
                            bbox_size=args.bbox_size,
                            rotation_x=args.rotation_x,
                            rotation_y=args.rotation_y,
                            uv_unwrap=args.uv_unwrap,
                            uv_rescale=args.uv_rescale,
                            use_existing_textures=use_textures)
    
    elev, azim, light_dirs = get_view_params(num_views=4)
    
    mesh_dict = mesh_dataset.render_images(image_resolution=args.image_resolution,
                                           dist=torch.tensor(args.camera_dist),
                                           elev=elev,
                                           azim=azim,
                                           fov=torch.tensor(args.camera_fov),
                                           light_dirs=light_dirs,
                                           use_textures=use_textures)
    
    view_strings = [view_angle_to_prompt(elev[i], azim[i]) for i in range(mesh_dict['mesh_images'].size(0))]
    
    if not args.production:
        torchvision.utils.save_image(mesh_dict['mesh_images'], f'{args.output_dir}/projected_mesh.png', padding=0)

    return mesh_dict, view_strings


def generate_avatar_image(args, mesh_dict, view_strings, use_view_prompt, prompt_masking_style, is_character_sheet,
                         input_images, output_name:str=""):
    output_image_name = f'{args.output_dir}/{output_name}'
    
    avatar_image_generator = AvatarImageGenerator(args, preloaded_models=None, device=args.device)
    generated_outputs = avatar_image_generator(
        mesh_dict,
        view_strings=view_strings,
        use_view_prompt=use_view_prompt,
        prompt_masking_style=prompt_masking_style,
        input_images=input_images,
        is_character_sheet=is_character_sheet,
        img2img_strength=args.img2img_strength,
    )
    output_images = generated_outputs['images']
    
    if not args.production and not output_name == "":
        torchvision.utils.save_image(output_images, output_image_name, padding=0) 

    return output_images, generated_outputs.get('diffusion_noise_init', None)


def interpolate_image(image0, image1, t):
    return (image0 * t) + (image1 * (1.0 - t))


#
# Generate character sheet with multiple views of the mesh in one image.
# This helps encourage multiview consistency.
#
def generate_initial_character_sheet(args, mesh_dict, view_strings):    
    # Add character_sheet_noise
    if args.character_sheet_noise > 0.0:
        noise_mask = (mesh_dict['mesh_depths'] > 0.05).float()
        input_noise = (torch.randn_like(mesh_dict['mesh_images']) * noise_mask) * args.character_sheet_noise
        mesh_dict['mesh_images'] = torch.clamp(interpolate_image(mesh_dict['mesh_images'], input_noise, 1.0 - args.character_sheet_noise), 0.0, 1.0)
    
    return generate_avatar_image(
        args=args,
        mesh_dict=mesh_dict,
        view_strings=view_strings,
        use_view_prompt=args.skip_character_sheet,
        prompt_masking_style=args.prompt_masking_style,
        is_character_sheet=not args.skip_character_sheet,
        input_images=mesh_dict['mesh_images'],
        output_name='depth2image.png',
    )


def setup_renderers(tsdf, use_pbr=False, bg_color=(0.0, 0.0, 0.0), device='cuda', bg_random_p=0.5):

    # PBR or albedo only
    material = PBRMaterial({
        "min_albedo": 0.03,
        "max_albedo": 0.8,
    }).to(device) if use_pbr else NoMaterial({}).to(device)

    # Setup renderer for optimization and testing
    bg = SolidColorBackground(dict(color=bg_color, random_aug=False, hls_color=True, s_range=(0.0, 0.01), random_aug_prob=bg_random_p)).to(device)
    bg_test = SolidColorBackground(dict(color=bg_color)).to(device)
    optimization_renderer = NVDiffRasterizer({"context_type": "cuda"}, geometry=tsdf, background=bg, material=material)
    test_renderer = NVDiffRasterizer({"context_type": "cuda"}, geometry=tsdf, background=bg_test, material=material)
    return dict(optimization=optimization_renderer, testing=test_renderer)


def direct_optimization_nvdiffrast(args, mesh_dict, target_images, target_masks, progress_callback=None):
    textured_mesh = mesh_dict['mesh']
    
    # writing temporary mesh
    output_mesh_basename = 'output_mesh.obj'
    output_texture_basename = 'tex_combined.png'
    tmp_mesh_dir = tempfile.mkdtemp(prefix='tmp_mesh_')
    tmp_mesh_filename = os.path.join(tmp_mesh_dir, output_mesh_basename)
    print('tmp_mesh_filename', tmp_mesh_filename)
    write_obj_with_texture(tmp_mesh_filename, output_texture_basename, textured_mesh)
    
    iter_num = args.num_sds_iterations
    guidance = args.guidance_sds
        
    implicit3d = setup_geometry3d(mesh_file=tmp_mesh_filename, geometry='custom_mesh', centering='none', scaling='none', material='pbr' if args.pbr_material else 'no_material')
    
    # Setup optimization and testing renderers for implicit representations
    renderers = setup_renderers(implicit3d, use_pbr=args.pbr_material, bg_random_p=1.0)

    optimization_output_dir = os.path.join(args.output_dir, 'optimization')
    os.makedirs(optimization_output_dir, exist_ok=True)
        
    optimizer3d = Optimizer3D(tsdf=implicit3d, renderers=renderers,
                              model_name=args.model_id,
                              controlnet_name=args.controlnet_name,
                              output_dir=optimization_output_dir,
                              distilled_encoder=args.distilled_encoder,
                              lambda_recon_reg=args.lambda_recon_reg,
                              lambda_albedo_smooth=args.lambda_albedo_smooth,
                              grad_clip=0.1,
                              save_img=0 if args.production else 100,
                              save_video=0 if args.production else 1000,
                              fix_geometry=True,
                              pretrained_dir=args.pretrained_dir,
                              guidance=guidance,
                              guidance_scale=args.guidance_scale,
                              cond_strength=args.cond_strength,
                              no_tqdm=args.no_tqdm,
                              camera_dist=args.SDS_camera_dist)
    
    implicit3d = optimizer3d.optimize_with_prompts(prompt=args.prompt, 
                                                   negative_prompt=args.n_prompt, 
                                                   num_iters=iter_num, 
                                                   textured_mesh=textured_mesh, 
                                                   fixed_target_images=target_images, 
                                                   fixed_target_masks=F.interpolate(target_masks, size=(512, 512), mode='bilinear'), 
                                                   fixed_target_azim=mesh_dict['azim'], 
                                                   fixed_target_elev=mesh_dict['elev'],
                                                   progress_callback=progress_callback)

    if not args.production:
        write_360_video_diffrast(renderers['testing'], output_filename=f"{optimization_output_dir}/{guidance}_final_rgb.gif")
        
        write_360_video_diffrast(renderers['testing'], output_filename=f"{optimization_output_dir}/{guidance}_final_rgb_up.gif", elev=-30)
        
        shutil.copyfile(f'{optimization_output_dir}/{guidance}_final_rgb.gif', f'{args.output_dir}/video360.gif')
    
    optimizer3d.export_mesh(optimization_output_dir, textured_mesh.textures.verts_uvs_padded().squeeze(0), textured_mesh.textures.faces_uvs_padded().squeeze(0))
    shutil.copyfile(f'{optimization_output_dir}/texture_kd.png', f'{args.output_dir}/texture_kd.png')
    shutil.copyfile(f'{optimization_output_dir}/output_mesh.mtl', f'{args.output_dir}/output_mesh.mtl')
    shutil.copyfile(f'{optimization_output_dir}/output_mesh.obj', f'{args.output_dir}/output_mesh.obj')

    if args.pbr_material:
        shutil.copyfile(f'{optimization_output_dir}/texture_metallic.png', f'{args.output_dir}/texture_metallic.png')
        shutil.copyfile(f'{optimization_output_dir}/texture_roughness.png', f'{args.output_dir}/texture_roughness.png')
        shutil.copyfile(f'{optimization_output_dir}/texture_nrm.png', f'{args.output_dir}/texture_nrm.png')

    
def main(args, progress_callback=None):
    args = setup(args)
    
    mesh_dict, view_strings = get_mesh_dict_and_view_strings(args, use_textures=args.refine)
    
    if args.guidance_sds == 'SDS_sd':
        output_images, diffusion_noise_init = generate_initial_character_sheet(args, mesh_dict, view_strings)
        output_images = F.interpolate(output_images, size=(512, 512), mode='bilinear')
        torchvision.utils.save_image(output_images[0:1], f'{args.output_dir}/depth2image_front.png', padding=0)
    else:
        output_images = None
    
    if progress_callback is not None:
        direct_optimization_nvdiffrast(args, mesh_dict, output_images, mesh_dict['mesh_masks'], progress_callback=progress_callback)

        
if __name__ == '__main__':
    args = parse_args()
    
    def progress_callback(image):
        torchvision.utils.save_image(image, f"{args.output_dir}/progress.png", padding=0)
        
    main(args, progress_callback=progress_callback)
