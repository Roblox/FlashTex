import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras
)

from mesh.render import draw_meshes
from optimization.guidance.set_loss_guidance import set_loss_guidance
from optimization.setup_geometry3d import setup_geometry3d
from utils.write_video import write_360_video_diffrast, fov_to_focal, get_ray_directions, get_rays

from optimization.utils.envlight_wrapper import CustomEnvLight

from threestudio.models.renderers.nvdiff_rasterizer import NVDiffRasterizer
from threestudio.models.materials.no_material import NoMaterial
from threestudio.models.materials.pbr_material import PBRMaterial
from threestudio.models.background.solid_color_background import SolidColorBackground

from models.vgg_loss import VGGLoss, VGGPerceptualLoss
from models.avatar_image_generator import (
    to_character_sheet,
    from_character_sheet
)

import envlight
import glob
import random

from scipy.spatial.transform import Rotation


class Optimizer3D:

    def __init__(self, tsdf, renderers, 
                 output_dir:str, 
                 distilled_encoder:str='./encoder_resnet4.pth',
                 preloaded_guidance=None, 
                 batch_size:int=4, 
                 image_resolution:int=512, 
                 rotation_y:float=0.0,
                 camera_fov:float=30.0,
                 camera_dist:float=5.0,
                 model_name='Lykon/DreamShaper',
                 controlnet_name=None,
                 guidance:str='SDS_sd',
                 guidance_scale:float=50.0,
                 min_noise_level:float=0.02,
                 max_noise_level:float=0.1,
                 grad_clip:float=0.0,
                 grad_center:bool=False,
                 weighting_strategy:str='fantasia3D',
                 sds_loss_style:str='standard',
                 lambda_recon_reg:float=1000.0,
                 save_img:int=100,
                 save_video:int=400,
                 progress_freq:int=50,
                 no_tqdm:bool=False,
                 fix_geometry:bool=True,
                 unwrap_uv:bool=False,
                 clip_tokenizer=None,
                 clip_text_model=None,
                 unet=None,
                 pretrained_dir:str=None,
                 device='cuda',
                 cond_strength=1.0,
                 lambda_albedo_smooth=0.0):

        self.device = device
        self.output_dir = output_dir

        self.tsdf = tsdf
        self.renderer = renderers['optimization']
        self.test_renderer = renderers['testing']

        self.batch_size = batch_size
        self.image_resolution = image_resolution

        self.rotation_y = rotation_y
        
        self.camera_fov = camera_fov
        self.camera_dist = camera_dist

        self.guidance = guidance
        self.guidance_scale = guidance_scale
        self.cond_strength = cond_strength
        
        self.lambda_recon_reg = lambda_recon_reg
        self.lambda_albedo_smooth = lambda_albedo_smooth

        self.save_image_freq = save_img
        self.save_video_freq = save_video
        self.progress_freq = progress_freq

        self.no_tqdm = no_tqdm
        
        self.fix_geometry = fix_geometry
        
        self.unwrap_uv = unwrap_uv
        
        if self.guidance == 'SDS_LightControlNet':
            # self.irr_maps_list = glob.glob("load/lights/sample/*.hdr")
            self.irr_maps_list = ["load/lights/studio_small_03_4k.hdr"]

            self.env_light_list = [CustomEnvLight(
                irr_map, scale=2.0
            ) for irr_map in self.irr_maps_list]

            self.env_light_cond_raw = CustomEnvLight(
                    "load/lights/studio_small_03_4k.hdr", scale=4.0
            )
            r = Rotation.from_euler('zyx', [180, 90+180, 0], degrees=True)
            rotation = torch.tensor(r.as_matrix()).float().to(device)
            self.env_light_cond_fixed = lambda *args, **kwargs: self.env_light_cond_raw(*args, **kwargs, rotation=rotation)
            
        
        self.env_light_regular = envlight.EnvLight(
            "load/lights/mud_road_puresky_1k.hdr", scale=2.0
        )
        
        if guidance == 'SDS_sd':
            self.env_light_cond = self.env_light_regular
            self.env_light_cond_fixed = self.env_light_regular
        
        try:
            pretrained_vgg_path = os.path.join(pretrained_dir, 'vgg', 'vgg19.pth')
            print(f'Loading pretrained VGG {pretrained_vgg_path}...')
            self.vgg_loss = VGGLoss(pretrained=False)
            self.vgg_loss.load_state_dict(torch.load(pretrained_vgg_path))
        except:
            print('Could not load pretrained VGG; loading from torch hub...')
            self.vgg_loss = VGGLoss()
        self.vgg_loss = self.vgg_loss.to(device)

        if guidance == 'SDS_LightControlNet':
            self.prepare_cond_renderer()
        
        self.loss_guidance = set_loss_guidance(guidance=guidance,
                                                min_noise_level=min_noise_level, 
                                                max_noise_level=max_noise_level, 
                                                model_name=model_name,
                                                controlnet_name=controlnet_name,
                                                distilled_encoder=distilled_encoder, 
                                                grad_clip=grad_clip, 
                                                grad_center=grad_center, 
                                                weighting_strategy=weighting_strategy, 
                                                sds_loss_style=sds_loss_style,
                                                clip_tokenizer=clip_tokenizer,
                                                clip_text_model=clip_text_model,
                                                unet=unet)
            
         

    def sample_character_sheet_cameras(self, sheet_size=2):
        num_view = sheet_size * sheet_size
        
        azim = np.linspace(0, 360, num=num_view+1)[:-1]
        elev = np.array([0, -15] * (num_view // 2))
        
        camera_dict = dict(dist=self.camera_dist, azim=torch.tensor(azim, dtype=torch.float32), elev=torch.tensor(elev, dtype=torch.float32), fov=self.camera_fov)
        
        
        return camera_dict
    
    def draw_implicit_batch_individual(self, batch_size, image_resolution, env_light=None):
        # Choose random camera parameters
        azim = (np.random.rand(batch_size)) * 360
        elev = np.random.randn(batch_size) * 15
        min_camera_fov = self.camera_fov - 25
        camera_fov_t = np.random.rand(batch_size)
        fov = self.camera_fov - ((self.camera_fov - min_camera_fov) * camera_fov_t)
        
        min_camera_dist = self.camera_dist - 1
        camera_dist_t = np.random.rand(batch_size)
        dist = self.camera_dist - ((self.camera_dist - min_camera_dist) * camera_dist_t)

        # Render image for chosen camera parameters
        camera_dict = dict(dist=dist, azim=azim, elev=elev, fov=fov)
        rendered_dict = self.draw_implicits(camera_dict, image_resolution=image_resolution, env_light=env_light, perturb=True)
        return rendered_dict

    def draw_implicit_batch_character_sheet(self, env_light):
        N = 4
        half_resolution = self.image_resolution // 2
        individual_rendered_dicts = []
        individual_rendered_dicts = self.draw_implicit_batch_individual(batch_size=self.batch_size*N, image_resolution=half_resolution, env_light=env_light)
        
        image_char_sheets = []
        normals_char_sheets = []
        sillhouettes_char_sheets = []
                            
        char_sheet_rendered_dicts = []
        for batch_idx in range(self.batch_size):
            start_idx = batch_idx * N
            end_idx = start_idx + N
            
            image_char_sheets.append(to_character_sheet(individual_rendered_dicts['images'][start_idx:end_idx], device=self.device))
            normals_char_sheets.append(to_character_sheet(individual_rendered_dicts['normals'][start_idx:end_idx], device=self.device))
            sillhouettes_char_sheets.append(to_character_sheet(individual_rendered_dicts['sillhouettes'][start_idx:end_idx], device=self.device))
            
        image_char_sheets = torch.cat(image_char_sheets, dim=0)
        normals_char_sheets = torch.cat(normals_char_sheets, dim=0)
        sillhouettes_char_sheets = torch.cat(sillhouettes_char_sheets, dim=0)
        
        return {
            'images': image_char_sheets,
            'normals': normals_char_sheets,
            'sillhouettes': sillhouettes_char_sheets,
            'camera': individual_rendered_dicts['camera'],
        }
    
    def optimize_with_prompts(self, prompt, negative_prompt, num_iters, textured_mesh=None, fixed_target_images=None, fixed_target_masks=None, fixed_target_azim=None, fixed_target_elev=None, progress_callback=None):
                    
        if self.guidance == 'SDS_LightControlNet':
            ref_camera_dict = dict(dist=self.camera_dist, azim=-fixed_target_azim, elev=-fixed_target_elev, fov=self.camera_fov)
        
            fixed_target_images = self.prepare_target_image(ref_camera_dict, prompt, negative_prompt, load=False)
        
        optimizer = torch.optim.Adam(self.tsdf.parameters(), lr=0.01)

        sds_loss_start = num_iters // 8
        
        print("Optimizing with prompts...")
        
        if self.no_tqdm:
            loop = range(num_iters)
        else:
            loop = tqdm(range(num_iters))
        for iter in loop:

            timestep_t = 1.0 - (float(iter - sds_loss_start) / float(num_iters - sds_loss_start))

            # Enable SDS after a warm-up period to init the texture from the fixed_target_images
            enable_sds_loss = iter >= sds_loss_start
            
            if self.guidance == 'SDS_LightControlNet':
                
                enable_illumination_loss = True

            else:
                enable_illumination_loss = False
            
            
            if enable_illumination_loss:
                # Randomize the lighting
                r = Rotation.from_euler('zyx', [np.random.rand(1).item() * 360] * 3, degrees=True)
                rotation = torch.tensor(r.as_matrix()).float().to(self.device)
                
                rand_env_light = random.choice(self.env_light_list)
                self.env_light_cond_random = lambda *args, **kwargs: rand_env_light(*args, **kwargs, rotation=rotation)
            
            if enable_illumination_loss:
                rendered_dict = self.draw_implicit_batch_individual(batch_size=self.batch_size, image_resolution=self.image_resolution, env_light=self.env_light_cond_random)
            else:
                rendered_dict = self.draw_implicit_batch_individual(batch_size=self.batch_size, image_resolution=self.image_resolution, env_light=self.env_light_regular)
                
            # rendered_dict = {k:v.detach() for k,v in rendered_dict.items()}
            rendered_image = rendered_dict['images']
            
            # Regularization
            loss_reg = 0
            
            if self.lambda_albedo_smooth > 0:
                # Albedo Smoothness Regularization
                loss_albedo_smooth = rendered_dict['albedo_grad'] * self.lambda_albedo_smooth + rendered_dict['roughness_grad'] * self.lambda_albedo_smooth * 10 + rendered_dict['metallic_grad'] * self.lambda_albedo_smooth * 10
            
                loss_reg = loss_reg + loss_albedo_smooth
            
            if self.lambda_recon_reg > 0:
                recon_azim = -fixed_target_azim + (torch.randn_like(fixed_target_azim) * 1.0)
                recon_elev = -fixed_target_elev + (torch.randn_like(fixed_target_elev) * 1.0)
                recon_fov = self.camera_fov
                recon_dist = self.camera_dist
                ref_image = fixed_target_images

                ref_sillhouette = fixed_target_masks
                
                # Render image for fixed target camera parameters
                recon_camera_dict = dict(dist=recon_dist, azim=recon_azim, elev=recon_elev, fov=recon_fov)
                
                num_view = len(fixed_target_azim)
        
                sheet_size = int(np.sqrt(num_view))

                res = 1024 // sheet_size
                
                recon_rendered_dict = self.draw_implicits(recon_camera_dict, image_resolution=res, env_light=self.env_light_cond_fixed)
                recon_rendered_image = recon_rendered_dict['images']                    
    
            
                def to224(x):
                    #return x
                    return F.interpolate(x, size=(224, 224), mode='bilinear')
                
                lambda_rgb = 1
                lambda_vgg = 1
                                
                view_loss_weights = 1.0
            
                # Compute loss and perform back propagation
                if lambda_rgb > 0.0:
                    loss_recon_reg_rgb = F.mse_loss(recon_rendered_image, ref_image, reduction='none')
                    loss_recon_reg_rgb = loss_recon_reg_rgb.reshape(loss_recon_reg_rgb.size(0), -1)
                    loss_recon_reg_rgb = (loss_recon_reg_rgb.mean(dim=1) * view_loss_weights).sum() * lambda_rgb
                else:
                    loss_recon_reg_rgb = 0.0
                
                if lambda_vgg > 0.0:
                    loss_recon_reg_vgg = (self.vgg_loss(to224(recon_rendered_image), to224(ref_image)).to(recon_rendered_image.device) * view_loss_weights).sum() * lambda_vgg
                else:
                    loss_recon_reg_vgg = 0.0
                
                loss_recon_reg = loss_recon_reg_rgb + loss_recon_reg_vgg
                
                loss_reg += loss_recon_reg * self.lambda_recon_reg
            else:
                recon_rendered_image = None
                ref_image = None
                                
            # Compute loss and perform back propagation
            optimizer.zero_grad()
            
            if enable_sds_loss:
                batch_size = rendered_image.size(0)
                azim = torch.from_numpy(rendered_dict['camera']['azim']).to(self.device)
                elev = torch.from_numpy(rendered_dict['camera']['elev']).to(self.device)
                dist = torch.from_numpy(rendered_dict['camera']['dist']).to(self.device)

                prompt_with_view_angle = [prompt + ', ' + self.view_angle_to_prompt(0.0, azim[i]) for i in range(batch_size)]
                if self.guidance == 'SDS_sd':
                    text_embeddings = self.loss_guidance.get_text_embeds(
                        prompt=prompt_with_view_angle, 
                        negative_prompt=[negative_prompt] * batch_size
                    )
                    
                    loss_sds = self.loss_guidance.train_step(
                        text_embeddings=text_embeddings, inputs=rendered_image, guidance_scale=self.guidance_scale, timestep_t=timestep_t
                    )
                elif self.guidance == 'SDS_LightControlNet':
                    text_embeddings = self.loss_guidance.get_text_embeds(
                        prompt=prompt_with_view_angle, 
                        negative_prompt=[negative_prompt] * batch_size
                    )
                    
                    img_cond = self.produce_cond_img(rendered_dict['camera'], env_light=self.env_light_cond_random)
                    img_cond = img_cond.transpose(0,3,1,2)


                    loss_sds = self.loss_guidance.train_step(
                        text_embeddings=text_embeddings, inputs=rendered_image, guidance_scale=self.guidance_scale, 
                        condition_image=img_cond, 
                        timestep_t=timestep_t / 2 + 0.5, 
                        # timestep_t = None, 
                        save_dir = self.output_dir if (iter+1) % 10 == 0 else None,
                        # cond_strength=self.cond_strength
                        cond_strength=timestep_t,
                        rescale_cfg=0.7

                    )
            
            else:
                loss_sds = 0.0
                    
            total_loss = loss_sds + loss_reg
            total_loss.backward()
            
            optimizer.step()
            
            if not self.no_tqdm:
                loop.set_description(f'Loss_Reg: {float(loss_reg):.4f}, Loss_SDS: {float(loss_sds):.4f}')
            
            # Save images
            iter_name = 'init' if iter == 0 else str(iter+1)
            if self.save_image_freq > 0 and (iter == 0 or (iter+1) % self.save_image_freq == 0 or (iter+1) == sds_loss_start):
                torchvision.utils.save_image(rendered_image, f"{self.output_dir}/{self.guidance}_{iter_name}.png", padding=0)
                if rendered_dict['images_diffuse'] is not None:
                    torchvision.utils.save_image(rendered_dict['images_diffuse'], f"{self.output_dir}/{self.guidance}_{iter_name}_diffuse.png")
                    torchvision.utils.save_image(rendered_dict['images_specular'], f"{self.output_dir}/{self.guidance}_{iter_name}_specular.png")

                    torchvision.utils.save_image(rendered_dict['images_albedo'], f"{self.output_dir}/{self.guidance}_{iter_name}_albedo.png")
                    torchvision.utils.save_image(rendered_dict['images_roughness'], f"{self.output_dir}/{self.guidance}_{iter_name}_roughness.png")
                    torchvision.utils.save_image(rendered_dict['images_metallic'], f"{self.output_dir}/{self.guidance}_{iter_name}_metallic.png")
                
                if rendered_dict['images_tangent'] is not None:
                    torchvision.utils.save_image(rendered_dict['images_tangent'], f"{self.output_dir}/{self.guidance}_{iter_name}_tangent.png")
                
                if recon_rendered_image is not None:
                    torchvision.utils.save_image(recon_rendered_image, f"{self.output_dir}/{self.guidance}_recon_{iter_name}.png", padding=0)
                if ref_image is not None:
                    torchvision.utils.save_image(ref_image, f"{self.output_dir}/{self.guidance}_ref_{iter_name}.png", padding=0)
               
            # Save video
            if self.save_video_freq > 0 and (iter+1) % self.save_video_freq == 0:
                write_360_video_diffrast(self.renderer, output_filename=f"{self.output_dir}/{self.guidance}_{iter+1}.gif")
                    
                self.render_with_rotate_light(self.renderer, output_filename=f"{self.output_dir}/{self.guidance}_{iter+1}.gif")
            
            # Progress
            if progress_callback is not None and (((iter+1) % self.progress_freq == 0) or (iter == num_iters - 1)):
                progress_callback(recon_rendered_image[0:1])

        
        return self.tsdf

    def draw_meshes(self, camera_dict, textured_mesh):
        R, T = look_at_view_transform(dist=camera_dict['dist'], elev=camera_dict['elev'], azim=camera_dict['azim'])
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=camera_dict['fov'])
        rendered_data = draw_meshes(
            meshes=textured_mesh.extend(self.batch_size),
            cameras=cameras,
            image_size=self.image_resolution,
            uv_masks=None,
            draw_flat = True,
            lights=None,
            training=False,
        )
        images = rendered_data['images']
        normals = rendered_data['normals'].clamp(0, 1).permute(0,3,1,2)
        sillhouettes = rendered_data['masks']
        return dict(images=images, normals=normals, sillhouettes=sillhouettes)


    def draw_implicits(self, camera_dict, image_resolution, env_light=None, **kwargs):
        R, T = look_at_view_transform(dist=camera_dict['dist'], elev=camera_dict['elev'], azim=camera_dict['azim'])
        if 'y_offset' in camera_dict:
            T[:,1] = T[:,1] + camera_dict['y_offset']
        
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=camera_dict['fov'])
        # c2w = cameras.get_world_to_view_transform().get_matrix().contiguous().to(self.device)  # Pytorch3d stores transposed matrix
        c2w = cameras.get_world_to_view_transform().inverse().get_matrix().transpose(1,2).contiguous().to(self.device)
        
        camera_positions = cameras.get_camera_center().to(self.device)

        light_distances = (torch.rand(len(cameras)) * (1.5 - 0.8) + 0.8).to(self.device)

        light_direction = F.normalize(camera_positions + torch.randn_like(camera_positions) * 1, dim=-1)
        light_positions = light_distances[..., None] * light_direction
        mvp_mtx = cameras.get_full_projection_transform().get_matrix().permute(0,2,1).contiguous().to(self.device)
        
        focal = fov_to_focal(camera_dict['fov'], self.image_resolution)
        if len(focal.shape) > 0:
            directions_all = []
            for i in range(focal.shape[0]):
                directions = get_ray_directions(self.image_resolution, self.image_resolution, focal[i]).to(self.device)
                directions_all.append(directions)
            directions = torch.stack(directions_all)
            # print(directions.shape)
        else:
            directions = get_ray_directions(self.image_resolution, self.image_resolution, focal).to(self.device)
        
        rays_o, rays_d = get_rays(directions, c2w)

        out = self.renderer(
            mvp_mtx=mvp_mtx, 
            camera_positions=camera_positions,
            light_positions=light_positions,
            height=image_resolution, width=image_resolution,
            c2w=c2w, rays_d=rays_d, env_bg=False, env_light=env_light,
            **kwargs
        )
        images = out['comp_rgb'].permute(0,3,1,2) # BCHW
        normals =  out['comp_normal'].permute(0,3,1,2)
        normals_viewspace = out['comp_normal_viewspace'].permute(0,3,1,2)
        sillhouettes =  out['opacity'].permute(0,3,1,2)
        
        
        if 'comp_rgb_diffuse' in out:
            images_diffuse = out['comp_rgb_diffuse'].permute(0,3,1,2) # BCHW
            images_specular = out['comp_rgb_specular'].permute(0,3,1,2) # BCHW

            images_albedo = out['comp_albedo'].permute(0,3,1,2) # BCHW
            images_roughness = out['comp_roughness'].permute(0,3,1,2) # BCHW
            images_metallic = out['comp_metallic'].permute(0,3,1,2) # BCHW
        else:
            images_diffuse, images_specular = None, None
            images_albedo, images_roughness, images_metallic = None, None, None
        
        if 'comp_tangent' in out:
            images_tangent = out['comp_tangent'].permute(0,3,1,2)
        else:
            images_tangent = None
        
        return dict(images=images, normals=normals, sillhouettes=sillhouettes, c2w=c2w, normals_viewspace=normals_viewspace,
                   images_diffuse=images_diffuse, images_specular=images_specular, 
                   images_mean_albedo=out["comp_rgb_mean_albedo"].permute(0,3,1,2) if "comp_rgb_mean_albedo" in out else None,
                   images_albedo=images_albedo, images_roughness=images_roughness, images_metallic=images_metallic,
                   albedo_grad=out['albedo_grad'] if "albedo_grad" in out else None,
                   roughness_grad=out['roughness_grad'] if "roughness_grad" in out else None,
                   metallic_grad=out['metallic_grad'] if "metallic_grad" in out else None,
                    tangent_grad=out['tangent_grad'] if "tangent_grad" in out else None,
                   camera=camera_dict,
                   images_tangent=images_tangent)

    @staticmethod
    def view_angle_to_prompt(elev, azim):
        azim = azim % 360
        if abs(azim - 180.0) < 90.0:
            return 'rear view'
        elif abs(azim) < 30.0 or abs(azim - 360) < 30:
            return 'front view'
        else:
            return 'side view'
        
    def export_mesh(self, output_dir, v_tex, t_tex_idx):
        from threestudio.models.exporters.mesh_exporter import MeshExporter
        from threestudio.utils.saving import SaverMixin
    

        self.exporter = MeshExporter({"save_name": "output_mesh",
                                     "context_type": "cuda",
                                     "texture_format": "png",
                                     "texture_size": 2048,
                                     "unwrap_uv": self.unwrap_uv,
                                     "save_normal": True,
                                    }, 
                                    geometry = self.tsdf,
                                    material = self.renderer.material,
                                    background = self.renderer.background,
                                    uv_dict = {
                                             "v_tex": v_tex.to(self.device),
                                             "t_tex_idx": t_tex_idx.to(self.device)
                                         })
        
        exporter_output = self.exporter()
        
        saving_util = SaverMixin()
        saving_util.set_save_dir(output_dir)
        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            save_func = getattr(saving_util, save_func_name)
            save_func(f"{out.save_name}", **out.params)
            
        torch.save(self.tsdf.state_dict(), output_dir + "/final_model.ckpt")

    def prepare_cond_renderer(self):
        material_m0r1 = PBRMaterial({"use_bump": False,
                        "min_metallic": 0.0,
                        "max_metallic": 0.0,
                        "min_roughness": 1.,
                        "max_roughness": 1.,
                        "min_albedo": 1.0,
                        "max_albedo": 1.0}).to(self.device)
        material_m5r5 = PBRMaterial({"use_bump": False,
                                "min_metallic": 0.5,
                                "max_metallic": 0.5,
                                "min_roughness": 0.5,
                                "max_roughness": 0.5,
                                "min_albedo": 1.0,
                                "max_albedo": 1.0}).to(self.device)
        material_m1r0 = PBRMaterial({"use_bump": False,
                                "min_metallic": 1.0,
                                "max_metallic": 1.0,
                                "min_roughness": 0.,
                                "max_roughness": 0.,
                                "min_albedo": 1.0,
                                "max_albedo": 1.0}).to(self.device)
        material_list = [material_m0r1, material_m5r5, material_m1r0]
        self.material_renderer_list = [NVDiffRasterizer({"context_type": "cuda"}, geometry=self.test_renderer.geometry, background=self.test_renderer.background, material=x) for x in material_list]


    def produce_cond_img(self, camera_dict, env_light=None, image_resolution = 512):
        R, T = look_at_view_transform(dist=camera_dict['dist'], elev=camera_dict['elev'], azim=camera_dict['azim'])
        if 'y_offset' in camera_dict:
            T[:,1] = T[:,1] + camera_dict['y_offset']
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=camera_dict['fov'])
        # cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=1, in_ndc=False, image_size=torch.tensor([512,512]).unsqueeze(0))
        c2w = cameras.get_world_to_view_transform().inverse().get_matrix().contiguous().to(self.device)

        camera_positions = cameras.get_camera_center().to(self.device)
        light_distances = (torch.rand(len(cameras)) * (1.5 - 0.8) + 0.8).to(self.device)
        light_direction = F.normalize(camera_positions + torch.randn_like(camera_positions) * 1, dim=-1)
        light_positions = light_distances[..., None] * light_direction
        mvp_mtx = cameras.get_full_projection_transform().get_matrix().permute(0,2,1).contiguous().to(self.device)

        focal = fov_to_focal(camera_dict['fov'], image_resolution)
        # print(focal)
        if len(focal.shape) > 0:
            directions_all = []
            for i in range(focal.shape[0]):
                directions = get_ray_directions(image_resolution, image_resolution, focal[i]).to(self.device)
                directions_all.append(directions)
            directions = torch.stack(directions_all)
            # print(directions.shape)
        else:
            directions = get_ray_directions(image_resolution, image_resolution, focal).to(self.device)

        rays_o, rays_d = get_rays(directions, c2w)


        img_conds = []
        for renderer in self.material_renderer_list:
            with torch.no_grad():
                out = renderer(
                    mvp_mtx=mvp_mtx, 
                    camera_positions=camera_positions,
                    light_positions=camera_positions,
                    height=image_resolution, width=image_resolution,
                    c2w=c2w, rays_d=rays_d, env_bg=False, env_light=env_light
                )
            img_conds.append(out['comp_rgb'].cpu().numpy())

        img_conds = [x[...,0] for x in img_conds]
        img_conds = np.stack(img_conds, axis=-1) # B x H x W x 3
        img_conds = img_conds.clip(0, 1)
        return img_conds


    def prepare_target_image(self, camera_dict, prompt, negative_prompt, load=False):
        num_view = len(camera_dict['azim'])
        
        sheet_size = int(np.sqrt(num_view))
        
        res = 1024 // sheet_size
        
        cond = self.produce_cond_img(camera_dict, self.env_light_cond_fixed, image_resolution=res)
        
        sheet_size = int(np.sqrt(cond.shape[0]))
        H, W = cond.shape[1], cond.shape[2]
        character_sheet = torchvision.utils.make_grid(torch.tensor(cond).permute(0,3,1,2).to(self.device), nrow=sheet_size, padding=0).unsqueeze(0)
        
        torchvision.utils.save_image(character_sheet, f"{self.output_dir}/ref_cond.png")
        
        control_image = character_sheet # 1 x 3 x H x W
        
        init_image = torch.cat([character_sheet[:,1:2]] * 3, dim=1)
        torchvision.utils.save_image(init_image, f"{self.output_dir}/ref_init.png")

        generator = torch.manual_seed(1)
        with torch.no_grad():
            image = self.loss_guidance.pipe(prompt, num_inference_steps=20, generator=generator, image=control_image, negative_prompt=negative_prompt).images[0]
            # image = pipe(prompt, num_inference_steps=20, generator=generator, image=init_image, negative_prompt=negative_prompt, control_image=control_image, strength=1.0).images[0]

        image.save(f"{self.output_dir}/ref_controlnet.png")
        
        ref_img = np.array(image) / 255
        ref_img = torch.tensor(ref_img).to(self.device).float()
        ref_img = ref_img.unsqueeze(0).permute(0, 3, 1, 2) # B x C x H x W
        
        # return from_character_sheet(ref_img, 4)
        
        images_list = torch.split(ref_img, H, dim=2)  # step=H, where 'H' is the height size
        images_list = [torch.split(img, W, dim=3) for img in images_list]  # step=W, where 'W' is the width size
        images_tensor = torch.cat([torch.cat(images) for images in images_list])
        
        
        return images_tensor
        
        
        
        
        
        
        
        