from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel,logging,CLIPProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDPMScheduler
import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import time

import os
import sys

from ldm.distilled_encoders import SmallResnetEncoder


class SDSLoss(nn.Module):
    def __init__(
            self,
            device,
            model_name='stabilityai/stable-diffusion-2-1-base',
            max_noise_level=0.98,
            min_noise_level=0.05,
            encoder_path=None,
            grad_clip=0,
            grad_center=False,
            weighting_strategy="fantasia3D",
            sds_loss_style='standard',
            clip_tokenizer=None,
            clip_text_model=None,
            unet=None,
            compile_unet=False,
        ):
        super().__init__()

        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * min_noise_level)
        self.max_step = int(self.num_train_timesteps * max_noise_level)
        self.use_distilled_encoder = encoder_path is not None

        self.sds_loss_style = sds_loss_style
        assert sds_loss_style in ['standard', 'image_mse_v1', 'image_mse_v2'], \
               f'Unknown SDS style {sds_loss_style}: standard | image_mse_v1 | image_mse_v2'

        print(f'loading stable diffusion with {model_name}...')

        # 0. Load the distilled image encoder model to encode images into latent space.
        if self.use_distilled_encoder:
            self.distilled_encoder = SmallResnetEncoder(in_channels=3, out_channels=4)
            self.distilled_encoder.load_state_dict(torch.load(encoder_path))
            self.distilled_encoder.to(self.device)

        # 1. Load the autoencoder model which will be used to encode and/or decode the latents.
        if not self.use_distilled_encoder or sds_loss_style in ['image_mse_v1', 'image_mse_v2']:
            self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float16).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        if clip_tokenizer:
            self.tokenizer = clip_tokenizer
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer", torch_dtype=torch.float16)
        if clip_text_model:
            self.text_encoder = clip_text_model
        else:
            self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=torch.float16).to(self.device)
        
        self.image_encoder = None
        self.image_processor = None

        # 3. The UNet model for generating the latents.
        if unet:
            self.unet = unet
        else:
            self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=torch.float16).to(self.device)
            
        if compile_unet:
            self.unet = torch.compile(self.unet, backend="inductor")

        # 4. Create a scheduler for inference
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'\t successfully loaded stable diffusion!')
        self.loss_time = 0
        self.unet_time = 0
        self.backward_time = 0
        
        self.grad_clip = grad_clip
        self.grad_center = grad_center
        
        self.weighting_strategy = weighting_strategy

    def get_text_embeds(self, prompt, negative_prompt=None):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

            # Do the same for unconditional embeddings
            if negative_prompt is None:
                uncond_input = self.tokenizer([''] * len(prompt), padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            else:
                uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    def train_step(self, text_embeddings, inputs, guidance_scale=100, timestep_t=None, save_dir=None):
        
        torch.cuda.synchronize(); t0 = time.time()

        # Interpolate to 512x512 prior to encoding (is this strictly necessary?)
        pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)

        # Encode images to latent space
        if self.sds_loss_style == 'standard':
            latents = self.encode_imgs(pred_rgb_512)
        elif self.sds_loss_style in ['image_mse_v1', 'image_mse_v2']:
            with torch.no_grad():
                latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if timestep_t is not None:
            t = torch.tensor([int(self.min_step + ((self.max_step - self.min_step) * timestep_t))], dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # predict the noise residual with unet
        with torch.no_grad():

            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input.to(torch.float16), t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        alpha_t = self.alphas[t]
        beta_t = 1 - alpha_t

        if self.sds_loss_style in ['standard', 'image_mse_v1']:
            w = alpha_t ** 0.5 * beta_t if self.weighting_strategy == 'fantasia3D' else beta_t

            grad = w * (noise_pred - noise)
            grad = grad - grad.mean(dim=1, keepdim=True) if self.grad_center else grad
            grad = grad.clamp(-self.grad_clip, self.grad_clip) if self.grad_clip > 0 else grad   # clip grad for stable training?

        if self.sds_loss_style == 'standard':
            target = (latents - grad).detach()
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / inputs.shape[0]

        elif self.sds_loss_style == 'image_mse_v1':
            image_pred = self.decode_latents((latents - grad)).detach()
            loss_sds = 0.05 * F.mse_loss(pred_rgb_512, image_pred, reduction="sum") / inputs.shape[0]

        elif self.sds_loss_style == 'image_mse_v2':
            z0_1step_pred = (latents - beta_t ** (0.5) * noise_pred) / alpha_t ** (0.5)
            # z0_1step_pred.clamp_(-1.0, 1.0)
            image_pred = self.decode_latents(z0_1step_pred).detach() 
            loss_sds = 0.0075 * F.mse_loss(pred_rgb_512, image_pred, reduction="sum") / inputs.shape[0]

        # loss_sds.backward()

        return loss_sds#.item()


    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs
    
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        if self.use_distilled_encoder:
            latents = self.distilled_encoder(imgs) * 0.18215
        else:
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, negative_prompt=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompt=negative_prompt) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

