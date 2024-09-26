import torch

def rescale_cfg(cond, uncond, cond_scale, multiplier=0.7):
    x_cfg = uncond + cond_scale * (cond - uncond)
    ro_pos = torch.std(cond, dim=(1,2,3), keepdim=True)
    ro_cfg = torch.std(x_cfg, dim=(1,2,3), keepdim=True)

    x_rescaled = x_cfg * (ro_pos / ro_cfg)
    x_final = multiplier * x_rescaled + (1.0 - multiplier) * x_cfg

    return x_final