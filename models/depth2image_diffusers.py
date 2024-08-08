import numpy as np
import torch

from diffusers import ControlNetModel
from diffusers import UniPCMultistepScheduler

from ldm.pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPromptMasksPipeline
from ldm.pipeline_controlnet import StableDiffusionControlNetPromptMasksPipeline


class Depth2Image_Diffusers():
    def __init__(
        self,
        model_id:str='runwayml/stable-diffusion-v1-5',
        controlnet_model_id="lllyasviel/control_v11f1p_sd15_depth",
        pretrained_dir:str=None,
        use_img2img:bool=False,
        compile:bool=False
    ) -> None:
        
        self.img2img = use_img2img
        if self.img2img:
            pipe_class = StableDiffusionControlNetImg2ImgPromptMasksPipeline
        else:
            pipe_class = StableDiffusionControlNetPromptMasksPipeline
        
        if pretrained_dir is not None:
            print(f'Loading pretrained {pretrained_dir}...')
            self.pipe = pipe_class.from_pretrained(
                pretrained_dir, 
                torch_dtype=torch.float16, 
                local_files_only=True,
                safety_checker=None,
                requires_safety_checker=False # fixme NSFW filter disabled for development
            ).to("cuda")
        else:
            controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16)
            self.pipe = pipe_class.from_pretrained(
                model_id, 
                controlnet=controlnet,
                torch_dtype=torch.float16, 
                safety_checker=None,
                requires_safety_checker=False # fixme NSFW filter disabled for development
            ).to("cuda")
        
        if compile:
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.controlnet.to(memory_format=torch.channels_last)
            print("Run torch compile")
            self.pipe.unet = torch.compile(self.pipe.unet, backend="inductor")
            self.pipe.controlnet = torch.compile(self.pipe.controlnet, backend="inductor")

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

    @torch.no_grad()
    def save_pretrained(self, output_dir:str):
        self.pipe.save_pretrained(save_directory=output_dir)
        
        
    @torch.no_grad()
    def process(
        self,
        input_depths,
        prompt, negative_prompt,
        image_resolution,
        ddim_steps, guess_mode, strength, scale, seed, eta,
        x_T=None,
        timesteps=None,
        rediffusion_image=None,
        rediffusion_extra_noise=0.0,
        output_dir='/output/decoded',
        save_decoded=False,
        img2img_strength=1.0,
        multiprompt_masks=None,
    ):
        input_depths = torch.cat([input_depths] * 3, dim=1)
        
        print('Generate depth shape:', input_depths.size())
        print('Prompts:', prompt)
        print('Negative prompts:', negative_prompt)
        
        generator = [torch.manual_seed(seed)] * input_depths.size(0)
        
        # use "prompt" or "prompt, additional_prompt" if additional_prompt is not empty
        assert(isinstance(prompt, list))
        # prompt = [(prompt[i] + ', ' + a_prompts[i]) if len(a_prompts[i]) > 0 else prompt[i] for i in range(len(prompt))]

        cross_attention_kwargs = dict(multiprompt_attention_masks=multiprompt_masks) if multiprompt_masks is not None else None

        if self.img2img:
            output_images = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(ddim_steps / img2img_strength),
                generator=generator,
                image=rediffusion_image,
                control_image=input_depths,
                guidance_scale=scale,
                eta=eta,
                controlnet_conditioning_scale=strength,
                strength=img2img_strength,
                cross_attention_kwargs=cross_attention_kwargs,
            ).images
        else:
            output_images = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=ddim_steps, 
                generator=generator,
                image=input_depths,
                guidance_scale=scale,
                eta=eta,
                controlnet_conditioning_scale=strength,
                cross_attention_kwargs=cross_attention_kwargs,
            ).images
        
        def from_pil(image):
            return torch.from_numpy(np.array(image).astype(np.float32) * (1.0 / 255.0)).permute(2,0,1)

        output_images = torch.stack([from_pil(image) for image in output_images], dim=0).cuda()

        return output_images, input_depths, None
        
