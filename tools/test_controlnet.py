import numpy as np
from PIL import Image
import os
import glob
from rich.progress import track

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

base_model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5" 
# controlnet_path = "/home/jovyan/data/kangled_vscode/checkpoints/controlnet_mvdream_sd15_68000"
controlnet_path = "kangled/lightcontrolnet"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

def disabled_safety_checker(images, clip_input):
    if len(images.shape)==4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False
pipe.safety_checker = disabled_safety_checker


prompt = ["material ball, brown leather, 3d asset ",
        "material ball, wooden, 3d asset",
        "material ball, steel, 3d asset"
]

negative_prompt = ["blur, bad quality"] * len(prompt)


generator = torch.manual_seed(1)

control_image_file = "load/examples/material_ball/011_cond.png"
control_image = [load_image(control_image_file)] * len(prompt)
images = pipe(prompt, num_inference_steps=20, generator=generator, image=control_image, negative_prompt=negative_prompt).images

# save images
for i, img in enumerate(images):
    img.save(f"output/lightcontrolnet_{i:03d}_out.png")