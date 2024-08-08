import math
import torch
import torch.nn.functional as F

from diffusers.models.attention import Attention


class PromptMasksAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        multiprompt_attention_masks=None,
        image_resolution=None,
        temb=None,
    ):

        use_split_cross_attention = encoder_hidden_states is not None and encoder_hidden_states.ndim == 4

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if use_split_cross_attention:
            _, batch_size, sequence_length, _ = encoder_hidden_states.shape
        else:
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
        inner_dim = hidden_states.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if use_split_cross_attention:

            batch_size, latent_image_size, num_attention_features_total = hidden_states.shape
            latent_image_downsample_factor = int(math.sqrt((image_resolution[0] * image_resolution[1]) // latent_image_size))
            latent_image_width, latent_image_height = [d // latent_image_downsample_factor for d in image_resolution]

            num_masks = multiprompt_attention_masks.shape[0]
            multiprompt_attention_masks = F.interpolate(
                multiprompt_attention_masks,
                size=(latent_image_height, latent_image_width),
                mode='nearest',
            ).reshape(num_masks, -1) > 0  # <<< reshape this to num_masks x (h*w)

            head_dim = num_attention_features_total // attn.heads

            key = attn.to_k(encoder_hidden_states).permute(1,0,2,3).flatten(1,2)
            value = attn.to_v(encoder_hidden_states).permute(1,0,2,3).flatten(1,2)

            multiprompt_attention_masks = multiprompt_attention_masks[None,None,:,None,:].repeat(2, attn.heads, 1, sequence_length, 1).flatten(2,3).permute(0,1,3,2)

            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=multiprompt_attention_masks, dropout_p=0.0, is_causal=False)
        
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



class PromptMasksAttnProcessor2_0_v0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        multiprompt_attention_masks=None,
        image_resolution=None,
        temb=None,
    ):

        use_split_cross_attention = encoder_hidden_states is not None and encoder_hidden_states.ndim == 4

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if use_split_cross_attention:
            _, batch_size, sequence_length, _ = encoder_hidden_states.shape
        else:
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
        inner_dim = hidden_states.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if use_split_cross_attention:

            batch_size, latent_image_size, num_attention_features_total = hidden_states.shape
            latent_image_downsample_factor = int(math.sqrt((image_resolution[0] * image_resolution[1]) // latent_image_size))
            latent_image_width, latent_image_height = [d // latent_image_downsample_factor for d in image_resolution]

            num_masks = multiprompt_attention_masks.shape[0]
            multiprompt_attention_masks = F.interpolate(
                multiprompt_attention_masks,
                size=(latent_image_height, latent_image_width),
                mode='nearest',
            ).reshape(num_masks, -1) > 0  # <<< reshape this to num_masks x (h*w)

            # Allocate space for cross-attention outputs; will iteratively fill in values for different masked regions
            head_dim = num_attention_features_total // attn.heads
            output_states = torch.zeros([batch_size, attn.heads, latent_image_size, head_dim]).to(hidden_states)

            # Loop through prompts and compute cross attention with each associated masked image
            for prompt_idx, encoded_prompts in enumerate(encoder_hidden_states):
                mask = multiprompt_attention_masks[prompt_idx]

                # Prepare attention components
                masked_query = query[:, mask]
                key = attn.to_k(encoded_prompts)
                value = attn.to_v(encoded_prompts)

                # Compute attention
                masked_query = masked_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                output_states[:, :, mask] = F.scaled_dot_product_attention(
                    masked_query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )

            hidden_states = output_states
        
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        # import pdb; pdb.set_trace()

        return hidden_states


