def set_loss_guidance(guidance:str, 
                      min_noise_level:float, 
                      max_noise_level:float, 
                      model_name:str,
                      controlnet_name:str,
                      distilled_encoder:str, 
                      grad_clip:float, 
                      grad_center:float, 
                      weighting_strategy:str, 
                      sds_loss_style:str,
                      clip_tokenizer=None,
                      clip_text_model=None,
                      unet=None,                  
                      device:str='cuda'):
    if guidance == 'SDS_sd':
        from optimization.guidance.SDS_sd import SDSLoss
        loss_guidance = SDSLoss(device=device, max_noise_level=max_noise_level, min_noise_level=min_noise_level, 
                                model_name=model_name, encoder_path=distilled_encoder, grad_clip=grad_clip, grad_center=grad_center,
                                weighting_strategy=weighting_strategy, sds_loss_style=sds_loss_style, clip_tokenizer=clip_tokenizer, clip_text_model=clip_text_model, unet=unet)
    elif guidance == 'SDS_LightControlNet':
        from optimization.guidance.SDS_control import SDSControlLoss
        loss_guidance = SDSControlLoss(device=device, max_noise_level=max_noise_level, min_noise_level=min_noise_level, 
                                controlnet_name=controlnet_name,
                                model_name=model_name, encoder_path=distilled_encoder, grad_clip=grad_clip, grad_center=grad_center,
                                weighting_strategy=weighting_strategy, clip_tokenizer=clip_tokenizer, clip_text_model=clip_text_model, unet=unet)
    else:
        print('Unknown guidance type:', guidance)
        exit()
    return loss_guidance
