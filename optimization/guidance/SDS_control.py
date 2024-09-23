from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel,logging,CLIPProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDPMScheduler, ControlNetModel, StableDiffusionControlNetPipeline
import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import time

import os
import sys

from ldm.distilled_encoders import SmallResnetEncoder

import torchvision

from optimization.guidance.utils import rescale_cfg as rescale_cfg_noise

class SDSControlLoss(nn.Module):
    def __init__(
            self,
            device,
            model_name='stabilityai/stable-diffusion-2-1-base',
            controlnet_name=None,
            max_noise_level=0.98,
            min_noise_level=0.02,
            encoder_path=None,
            grad_clip=0,
            grad_center=False,
            weighting_strategy="fantasia3D",
            clip_tokenizer=None,
            clip_text_model=None,
            unet=None,
            compile_unet=False
        ):
        super().__init__()

        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * min_noise_level)
        self.max_step = int(self.num_train_timesteps * max_noise_level)
        self.use_distilled_encoder = encoder_path is not None

        print(f'loading stable diffusion with {model_name}...')

        # 0. Load the distilled image encoder model to encode images into latent space.
        if self.use_distilled_encoder:
            self.distilled_encoder = SmallResnetEncoder(in_channels=3, out_channels=4)
            self.distilled_encoder.load_state_dict(torch.load(encoder_path))
            self.distilled_encoder.to(self.device)

        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        else:
            self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device)    # <~~~ TODO: This also includes decoder; check if needed

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
        
        
        # 5. Load the controlnet
        self.controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=torch.float16).to(self.device)

        self.pipe = StableDiffusionControlNetPipeline(
            vae=AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float16).to(self.device), 
            text_encoder=self.text_encoder, tokenizer=self.tokenizer,
            unet=self.unet, controlnet=self.controlnet,
            scheduler=self.scheduler, requires_safety_checker=False,
            safety_checker=None, feature_extractor=None
        )
        self.pipe.enable_xformers_memory_efficient_attention()

        self.pipe.enable_model_cpu_offload()

        # def disabled_safety_checker(images, clip_input):
        #     if len(images.shape)==4:
        #         num_images = images.shape[0]
        #         return images, [False]*num_images
        #     else:
        #         return images, False
        # self.pipe.safety_checker = disabled_safety_checker
        
        print(f'\t successfully loaded stable diffusion controlnet!')
        self.loss_time = 0
        self.unet_time = 0
        self.backward_time = 0
        
        self.grad_clip = grad_clip
        self.grad_center = grad_center
        
        self.weighting_strategy = weighting_strategy
        
        # self.controlnet_conditioning_scale = 1.0

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


    def train_step(self, text_embeddings, inputs, guidance_scale=100, condition_image=None, save_dir=None, timestep_t=None, cond_strength=1.0, rescale_cfg=1.0):
        
        if save_dir is not None:
            if condition_image is not None:
                torchvision.utils.save_image(torch.tensor(condition_image), f"{save_dir}/progress_SDS_img_cond.png")
                torchvision.utils.save_image(inputs, f"{save_dir}/progress_SDS_rendered.png")
        
        torch.cuda.synchronize(); t0 = time.time()

        batch_size = inputs.shape[0]
        
        # Interpolate to 512x512 prior to encoding (is this strictly necessary?)
        pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)

        # Encode images to latent space
        if self.use_distilled_encoder:
            latents = self.fast_encode_imgs(pred_rgb_512)
        else:
            latents = self.encode_imgs(pred_rgb_512)            

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if timestep_t is not None:
            t = torch.tensor([int(self.min_step + ((self.max_step - self.min_step) * timestep_t))], dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():

            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            
            if condition_image is not None:
                # controlnet conditioning
                
                condition_image = torch.tensor(condition_image).to(torch.float16).to(self.device)
                condition_image = torch.cat([condition_image] * 2)
                
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input.to(torch.float16),
                    t.repeat(latent_model_input.shape[0]).to(self.controlnet.device),
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=condition_image,
                    return_dict=False,
                )

                down_block_res_samples = [
                    down_block_res_sample * cond_strength
                    for down_block_res_sample in down_block_res_samples
                ]
                mid_block_res_sample *= cond_strength

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input.to(torch.float16),
                    t,
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            else:            
                noise_pred = self.unet(latent_model_input.to(torch.float16), t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        
        if rescale_cfg == 1.0:
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = rescale_cfg_noise(noise_pred_text, noise_pred_uncond, guidance_scale, rescale_cfg)

        if self.weighting_strategy == 'fantasia3D':
            w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        elif self.weighting_strategy == 'SDS':
            w = (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        if self.grad_center:
            # grad [B,C,H,W]
            assert grad.shape[1] == 4
            grad_center = grad.mean(dim=1, keepdim=True)
            grad = grad - grad_center
            # grad_var = grad.var(dim=1, keepdim=True)
            # grad = grad / torch.sqrt(grad_var)
        
        # clip grad for stable training?
        if self.grad_clip > 0:
            grad = grad.clamp(-self.grad_clip, self.grad_clip)
        

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # latents.backward(gradient=grad, retain_graph=True)
        
        # Calculate a meaningful loss formulation
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / inputs.shape[0]
        # loss_sds.backward()

        return loss_sds

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

    def fast_encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        latents = self.distilled_encoder(imgs) * 0.18215

        return latents
    
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

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
 