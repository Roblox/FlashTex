import os
import sys

import torch
import numpy as np
from PIL import Image

from models.depth2image_diffusers import Depth2Image_Diffusers



def to_character_sheet(images, device):
    N, C, H, W = images.size()
    assert(N <= 4)#, 'Character sheets have a max of 4 views')
    
    if N > 2:
        sheet = torch.zeros((1, C, H*2, W*2), dtype=torch.float32, device=device)
    else:
        sheet = torch.zeros((1, C, H, W*2), dtype=torch.float32, device=device)
        
    for image_idx in range(N):
        dst_x = (image_idx % 2) * W
        dst_y = (image_idx // 2) * H
        sheet[0,:,dst_y:dst_y+H,dst_x:dst_x+W] = images[image_idx]
    
    return sheet

    
def from_character_sheet(sheet, N):
    _, C, H2, W2 = sheet.size()
    if N > 2:
        H = H2//2
    else:
        H = H2
    W = W2//2
    images = []
    for image_idx in range(N):
        src_x = (image_idx % 2) * W
        src_y = (image_idx // 2) * H
        image = sheet[:,:,src_y:src_y+H,src_x:src_x+W]
        images.append(image)
    images = torch.cat(images, dim=0)
        
    return images
    
    
class AvatarImageGenerator:

    def __init__(self, args, preloaded_models, device=torch.device('cuda'), compile:bool=False):

        self.device = device
        self.args = args
        self.output_dir = args.output_dir
        self.compile = compile
        
        refine_or_not = 'refine_' if args.refine else ''
        model_id_or_not = f'_{args.model_id}' if args.model_id else ''
        diffusion_model_name = f'{refine_or_not}depth2image{model_id_or_not}'
        
        if preloaded_models is not None and diffusion_model_name in preloaded_models:
            self.diffusion_model = preloaded_models[diffusion_model_name]
        else:
            use_img2img = not args.disable_img2img
            print('Using Img2Img:', use_img2img)
            pretrained_dir = args.pretrained_dir
            if pretrained_dir is not None and pretrained_dir != '':
                pretrained_dir = os.path.join(pretrained_dir, 'diffusers', args.model_id)
            self.diffusion_model = self.create_diffusion_model(model_id=args.model_id, use_img2img=use_img2img, pretrained_dir=pretrained_dir)

    def create_diffusion_model(self, model_id:str, use_img2img:bool, pretrained_dir:str):
        args = self.args
        diffusion_model_name = f'depth2image_{model_id}'
        print(f'\nAvatarImageGenerator Using diffusion model: {diffusion_model_name} (img2img: {use_img2img})')

        model_args = dict(
            model_id=model_id,
            use_img2img=use_img2img,
            pretrained_dir=pretrained_dir,
            compile=self.compile
        )
        
        model = Depth2Image_Diffusers(**model_args)

        return model

    def __call__(
        self,
        mesh_dict,
        view_strings,
        use_view_prompt=False,
        prompt_masking_style='global',
        input_images:torch.tensor=None,
        is_character_sheet:bool=False,
        img2img_strength=1,
    ):
        
        args = self.args
        mesh_depths = mesh_dict['mesh_depths']
        num_views = mesh_depths.size(0)
        prompts = [args.prompt] * num_views
        
        if is_character_sheet:
            input_images = to_character_sheet(input_images, device=args.device)
            mesh_depths = to_character_sheet(mesh_depths, device=args.device)
            prompts = [prompts[0]]
        print('Prompts:', prompts)

        prompt_masking_style = prompt_masking_style if is_character_sheet else None

        if prompt_masking_style == 'global':
            if use_view_prompt:
                a_prompts = view_strings
            else:
                a_prompts = [args.a_prompt] * num_views
            prompts = [(prompt + ', ' + a_prompt) if a_prompt else prompt for prompt, a_prompt in zip(prompts, a_prompts)]
            
            if use_view_prompt:
                negative_prompts = []
                for view_string in view_strings:
                    if ('back' in view_string or 'rear' in view_string):  # <~~~~ TODO: Make this check more robust
                        negative_prompts.append(args.negative_prompt_back)
                    else:
                        negative_prompts.append(args.n_prompt)
            else:
                negative_prompts = [args.n_prompt] * mesh_depths.size(0)
                
            multiprompt_masks = None

            print('View prompts:', prompts)
            print('Negative view prompts:', negative_prompts)
            
        elif prompt_masking_style == 'front_back':

            # Construct masks
            mask_width = mesh_depths.shape[-1]
            zeros_mask = torch.zeros_like(mesh_depths[..., :mask_width//2])
            ones_mask = torch.ones_like(mesh_depths[..., :mask_width//2])

            front_mask = torch.cat([ones_mask, zeros_mask], dim=-1)
            back_mask = torch.cat([zeros_mask, ones_mask], dim=-1)
            multiprompt_masks = torch.cat([front_mask, back_mask], dim=0)  # dim = (2 x 1 x H x W)

            # Construct prompts
            prompts = [
                'front view of ' + prompts[0] + ', front view',
                'back view of ' + prompts[0] + ', back view',
            ]
            negative_prompts = [
                args.n_prompt + ', back view',
                args.n_prompt + ', front view'
            ]

        elif prompt_masking_style == 'front_back_localized':

            # Construct masks
            mask_width = mesh_depths.shape[-1]
            zeros_mask = torch.zeros_like(mesh_depths[..., :mask_width//2])
            ones_mask = torch.ones_like(mesh_depths[..., :mask_width//2])

            front_mask = torch.cat([ones_mask, zeros_mask], dim=-1) * (mesh_depths > 0.0).float()
            back_mask = torch.cat([zeros_mask, ones_mask], dim=-1) * (mesh_depths > 0.0).float()
            background_mask = torch.ones_like(mesh_depths) - front_mask - back_mask
            multiprompt_masks = torch.cat([background_mask, front_mask, back_mask], dim=0)  # dim = (3 x 1 x H x W)

            # Construct prompts
            prompts = [
                'a nice background image',
                'front view of ' + prompts[0] + ', front view',
                'back view of ' + prompts[0] + ', back view',
            ]
            negative_prompts = [
                args.n_prompt,
                args.n_prompt + ', back view',
                args.n_prompt + ', front view'
            ]

        elif prompt_masking_style is not None:
            sys.exit(f'Invalid prompt_masking_style: {prompt_masking_style}')

        else:
            negative_prompts = None
            multiprompt_masks = None

        rediffusion_image = input_images
        timesteps = args.ddim_steps if rediffusion_image is None else ((args.ddim_steps * 2) // 3)

        diffusion_args = dict(
            prompt=prompts,
            negative_prompt=negative_prompts,
            image_resolution=args.image_resolution,
            ddim_steps=args.ddim_steps,
            guess_mode=False,
            strength=args.strength,
            scale=args.scale,
            seed=args.seed,
            eta=args.eta,
            timesteps=timesteps,
            rediffusion_image=rediffusion_image,
            output_dir = self.output_dir,
        )
        
        if not args.disable_img2img:
            diffusion_args['img2img_strength'] = img2img_strength

        diffusion_args['input_depths'] = mesh_depths
        diffusion_args['multiprompt_masks'] = multiprompt_masks
            
        output_images, input_control, diffusion_noise_init = self.diffusion_model.process(**diffusion_args)

        if is_character_sheet:
            output_images = from_character_sheet(output_images, N=num_views)
        
        return dict(
            images=output_images,
            control=input_control,
            # additional_prompts=additional_prompts,
            negative_prompts=negative_prompts,
            diffusion_noise_init=diffusion_noise_init,
        )
