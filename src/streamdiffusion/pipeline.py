import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal

import numpy as np
import PIL.Image
import torch
from diffusers import LCMScheduler, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from streamdiffusion.image_filter import SimilarImageFilter


class StreamDiffusion:
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        self.cfg_type = cfg_type

        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (
                    self.denoising_steps_num + 1
                ) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = (
                    2 * self.denoising_steps_num * self.frame_bff_size
                )
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        # Check if the pipeline is an SDXL model
        self.is_xl = getattr(pipe, 'is_xl', False)
        if self.is_xl:
            print("[StreamDiffusion.__init__] SDXL model detected.")
        else:
            print("[StreamDiffusion.__init__] Non-SDXL model detected (or is_xl attribute not found on input pipe).")

        self.inference_time_ema = 0

    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, Dict[str, torch.Tensor]
        ] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
    ) -> None:
        self.generator = generator
        self.generator.manual_seed(seed)
        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True

        # SDXL-specific attributes to be populated if is_xl is True
        self.sdxl_pooled_prompt_embeds = None
        self.sdxl_negative_pooled_prompt_embeds = None
        self.sdxl_add_time_ids = None

        if hasattr(self, 'is_xl') and self.is_xl:
            # Logic for SDXL prompt encoding
            # Note: encode_prompt for SDXL returns (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
            prompt_embeds_main, negative_prompt_embeds_main, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )

            self.sdxl_pooled_prompt_embeds = pooled_prompt_embeds.to(self.device) if pooled_prompt_embeds is not None else None
            if do_classifier_free_guidance:
                self.sdxl_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(self.device) if negative_pooled_prompt_embeds is not None else None
            else: # Ensure it's None if not CFG, consistent with initialization
                self.sdxl_negative_pooled_prompt_embeds = None

            # Prepare main prompt embeddings (self.prompt_embeds) for UNet's encoder_hidden_states
            _cond_prompt_embeds_main_xl = prompt_embeds_main.repeat(self.batch_size, 1, 1)
            if do_classifier_free_guidance:
                # Determine batch size for unconditional embeddings based on original logic
                batch_size_for_uncond = self.batch_size if (self.use_denoising_batch and self.cfg_type == "full") else self.frame_bff_size
                _uncond_prompt_embeds_main_xl = negative_prompt_embeds_main.repeat(batch_size_for_uncond, 1, 1)

                if self.cfg_type == "initialize" or self.cfg_type == "full":
                    self.prompt_embeds = torch.cat([_uncond_prompt_embeds_main_xl, _cond_prompt_embeds_main_xl], dim=0)
                elif self.cfg_type == "self":
                    # For 'self' CFG, StreamDiffusion's UNet call only gets conditional embeds.
                    # Unconditional part is mixed later using stock_noise. This is standard for main embeds.
                    self.prompt_embeds = _cond_prompt_embeds_main_xl
                    # Storing uncond main embeds separately in case stock_noise needs them later for SDXL (though unlikely for this patch scope)
                    self.sdxl_uncond_main_prompt_embeds_for_self_cfg = _uncond_prompt_embeds_main_xl 
                else: # cfg_type == "none" or other unhandled CFG for SDXL
                    self.prompt_embeds = _cond_prompt_embeds_main_xl
            else: # No classifier-free guidance
                self.prompt_embeds = _cond_prompt_embeds_main_xl

            # Generate and store add_time_ids for SDXL
            # ==== BEGIN SDXL DIAGNOSTIC LOGGING (StreamDiffusion.prepare) ====
            print("[StreamDiffusion.prepare] Checking self.pipe.text_encoder_2 before _get_add_time_ids call...")
            if hasattr(self.pipe, 'text_encoder_2') and self.pipe.text_encoder_2 is not None:
                print("[StreamDiffusion.prepare] self.pipe.text_encoder_2 exists.")
                if hasattr(self.pipe.text_encoder_2, 'config') and self.pipe.text_encoder_2.config is not None:
                    print("[StreamDiffusion.prepare] self.pipe.text_encoder_2.config exists.")
                    if hasattr(self.pipe.text_encoder_2.config, 'projection_dim'):
                        print(f"[StreamDiffusion.prepare] self.pipe.text_encoder_2.config.projection_dim: {self.pipe.text_encoder_2.config.projection_dim}")
                        print(f"[StreamDiffusion.prepare] Type of projection_dim: {type(self.pipe.text_encoder_2.config.projection_dim)}")
                    else:
                        print("[StreamDiffusion.prepare] self.pipe.text_encoder_2.config does NOT have attribute 'projection_dim'.")
                else:
                    print("[StreamDiffusion.prepare] self.pipe.text_encoder_2 does NOT have attribute 'config' or it is None.")
            else:
                print("[StreamDiffusion.prepare] self.pipe does NOT have attribute 'text_encoder_2' or it is None.")
            print("[StreamDiffusion.prepare] Also checking self.unet.config.cross_attention_dim as fallback...")
            if hasattr(self.pipe, 'unet') and hasattr(self.pipe.unet, 'config') and hasattr(self.pipe.unet.config, 'cross_attention_dim'):
                 print(f"[StreamDiffusion.prepare] self.pipe.unet.config.cross_attention_dim: {self.pipe.unet.config.cross_attention_dim}")
            else:
                 print("[StreamDiffusion.prepare] Could not retrieve self.pipe.unet.config.cross_attention_dim")
            print("[StreamDiffusion.prepare] Checking self.pipe.unet.config.addition_time_embed_dim...")
            if hasattr(self.pipe, 'unet') and hasattr(self.pipe.unet, 'config') and hasattr(self.pipe.unet.config, 'addition_time_embed_dim'):
                 print(f"[StreamDiffusion.prepare] self.pipe.unet.config.addition_time_embed_dim: {self.pipe.unet.config.addition_time_embed_dim}")
                 print(f"[StreamDiffusion.prepare] Type of addition_time_embed_dim: {type(self.pipe.unet.config.addition_time_embed_dim)}")
            else:
                 print("[StreamDiffusion.prepare] Could not retrieve self.pipe.unet.config.addition_time_embed_dim or it's missing.")
            # ==== END SDXL DIAGNOSTIC LOGGING (StreamDiffusion.prepare) ====
            
            # Determine the projection_dim to pass
            proj_dim_to_pass = None
            if hasattr(self.pipe, 'text_encoder_2') and self.pipe.text_encoder_2 is not None and \
               hasattr(self.pipe.text_encoder_2, 'config') and self.pipe.text_encoder_2.config is not None and \
               hasattr(self.pipe.text_encoder_2.config, 'projection_dim'):
                proj_dim_to_pass = self.pipe.text_encoder_2.config.projection_dim
            
            print(f"[StreamDiffusion.prepare] Attempting to pass text_encoder_projection_dim: {proj_dim_to_pass} (type: {type(proj_dim_to_pass)}) to _get_add_time_ids")

            _add_time_ids_val = self.pipe._get_add_time_ids(
                original_size=(self.height, self.width),
                crops_coords_top_left=(0, 0),
                target_size=(self.height, self.width),
                dtype=prompt_embeds_main.dtype,
                text_encoder_projection_dim=proj_dim_to_pass # Explicitly pass it
            ) # Original call
            self.sdxl_add_time_ids = _add_time_ids_val.to(self.device) if _add_time_ids_val is not None else None
        else:
            # Original non-SDXL prompt encoding logic
            encoder_output = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )
            self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

            if do_classifier_free_guidance and (self.cfg_type == "initialize" or self.cfg_type == "full"):
                batch_size_for_uncond = self.batch_size if (self.use_denoising_batch and self.cfg_type == "full") else self.frame_bff_size
                uncond_prompt_embeds = encoder_output[1].repeat(batch_size_for_uncond, 1, 1)
                self.prompt_embeds = torch.cat(
                    [uncond_prompt_embeds, self.prompt_embeds], dim=0
                )   

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        noisy_samples = (
            self.alpha_prod_t_sqrt[t_index] * original_samples
            + self.beta_prod_t_sqrt[t_index] * noise
        )
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        # TODO: use t_list to select beta_prod_t_sqrt
        if idx is None:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch
            ) / self.alpha_prod_t_sqrt
            denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch
            ) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = (
                self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
            )

        return denoised_batch

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.isnan(x_t_latent).any() or torch.isinf(x_t_latent).any():
            print(f"[StreamDiffusion.unet_step] WARNING: Input x_t_latent to unet_step contains NaN/inf values!")
            print(f"[StreamDiffusion.unet_step] x_t_latent (input) min: {x_t_latent.min().item() if x_t_latent.numel() > 0 else 'N/A'}, max: {x_t_latent.max().item() if x_t_latent.numel() > 0 else 'N/A'}, mean: {x_t_latent.mean().item() if x_t_latent.numel() > 0 else 'N/A'}")

        # Ensure t_list is a tensor
        if not isinstance(t_list, torch.Tensor):
            t_list = torch.tensor(t_list, device=self.device, dtype=torch.long)

        # Adjust t_list if using denoising batch and frame_buffer_size > 1
        # This ensures t_list matches the batch dimension of x_t_latent *before* CFG processing.
        if self.use_denoising_batch and hasattr(self, 'frame_bff_size') and self.frame_bff_size > 1:
            # x_t_latent (input to unet_step) has batch_size = DSN * FBS.
            # t_list (input to unet_step) has batch_size = DSN.
            # We need t_list to be DSN * FBS for the UNet if no CFG, or for CFG logic to build upon.
            if t_list.shape[0] == self.denoising_steps_num: # Check if it's the original DSN-length t_list
                t_list = t_list.repeat_interleave(self.frame_bff_size)

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            if torch.isnan(x_t_latent).any() or torch.isinf(x_t_latent).any():
                print(f"[StreamDiffusion.unet_step] WARNING: x_t_latent (conditional) contains NaN/inf values before torch.cat!")
                print(f"[StreamDiffusion.unet_step] x_t_latent min: {x_t_latent.min().item() if x_t_latent.numel() > 0 else 'N/A'}, max: {x_t_latent.max().item() if x_t_latent.numel() > 0 else 'N/A'}, mean: {x_t_latent.mean().item() if x_t_latent.numel() > 0 else 'N/A'}")
            
            if self.cfg_type == "full" and (torch.isnan(self.empty_latent).any() or torch.isinf(self.empty_latent).any()):
                print(f"[StreamDiffusion.unet_step] WARNING: self.empty_latent (source for uc) contains NaN/inf values!")
                print(f"[StreamDiffusion.unet_step] self.empty_latent min: {self.empty_latent.min().item() if self.empty_latent.numel() > 0 else 'N/A'}, max: {self.empty_latent.max().item() if self.empty_latent.numel() > 0 else 'N/A'}, mean: {self.empty_latent.mean().item() if self.empty_latent.numel() > 0 else 'N/A'}")

            x_t_latent_plus_uc = torch.cat([x_t_latent, self.empty_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

            # Determine the batch size of x_t_latent_plus_uc, which is what UNet will see
            unet_input_batch_size = x_t_latent_plus_uc.shape[0]

        added_cond_kwargs = None
        if hasattr(self, 'is_xl') and self.is_xl:
            # Ensure pooled embeds and time_ids are on the correct device and dtype
            # These should be pre-set by the prepare() method
            _current_text_embeds = self.sdxl_pooled_prompt_embeds.to(device=self.device, dtype=self.dtype)
            _current_add_time_ids = self.sdxl_add_time_ids.to(device=self.device, dtype=self.dtype)

            # Handle CFG for SDXL pooled embeds and time_ids
            if self.guidance_scale > 1.0 and (self.cfg_type == "full" or self.cfg_type == "initialize"):
                if not hasattr(self, 'sdxl_negative_pooled_prompt_embeds') or self.sdxl_negative_pooled_prompt_embeds is None:
                    # This case should ideally be prevented by checks in prepare() or __init__ if CFG is enabled for SDXL
                    print("[StreamDiffusion.unet_step] WARNING: SDXL CFG is active, but sdxl_negative_pooled_prompt_embeds is missing. Falling back to using positive embeds for negative.")
                    _current_negative_text_embeds = self.sdxl_pooled_prompt_embeds.to(device=self.device, dtype=self.dtype) # Fallback
                else:
                    _current_negative_text_embeds = self.sdxl_negative_pooled_prompt_embeds.to(device=self.device, dtype=self.dtype)
                
                _current_text_embeds = torch.cat([_current_negative_text_embeds, _current_text_embeds], dim=0)
                
                # num_repeats_for_cfg is the original batch size of x_t_latent (before CFG duplication by unet_step)
                num_repeats_for_cfg = x_t_latent.shape[0]
                _current_text_embeds = _current_text_embeds.repeat_interleave(num_repeats_for_cfg, dim=0)

                if _current_add_time_ids.ndim == 1:
                    _current_add_time_ids = _current_add_time_ids.unsqueeze(0) 
                # For CFG, add_time_ids are repeated to match the doubled batch size of text_embeds
                _current_add_time_ids = _current_add_time_ids.repeat(2 * num_repeats_for_cfg, 1)

            else: # Not using CFG for SDXL specific embeddings
                if _current_text_embeds.ndim == 1:
                    _current_text_embeds = _current_text_embeds.unsqueeze(0) 
                _current_text_embeds = _current_text_embeds.repeat(unet_input_batch_size, 1)

                if _current_add_time_ids.ndim == 1:
                    _current_add_time_ids = _current_add_time_ids.unsqueeze(0)
                _current_add_time_ids = _current_add_time_ids.repeat(unet_input_batch_size, 1)

            added_cond_kwargs = {"text_embeds": _current_text_embeds, "time_ids": _current_add_time_ids}

        # ==== BEGIN SDXL DIAGNOSTIC LOGGING (StreamDiffusion.unet_step) ====
        if hasattr(self, 'is_xl') and self.is_xl:
            print(f"[StreamDiffusion.unet_step] SDXL Mode Active")
            print(f"  Input x_t_latent batch: {x_t_latent.shape[0]}")
            print(f"  UNet input batch size (x_t_latent_plus_uc): {unet_input_batch_size}")
            print(f"  Guidance Scale: {self.guidance_scale}, CFG Type: {self.cfg_type}")
            print(f"  Shape of self.sdxl_pooled_prompt_embeds (from prepare): {self.sdxl_pooled_prompt_embeds.shape if hasattr(self, 'sdxl_pooled_prompt_embeds') and self.sdxl_pooled_prompt_embeds is not None else 'Not Found'}")
            print(f"  Shape of self.sdxl_add_time_ids (from prepare): {self.sdxl_add_time_ids.shape if hasattr(self, 'sdxl_add_time_ids') and self.sdxl_add_time_ids is not None else 'Not Found'}")
            if hasattr(self, 'sdxl_negative_pooled_prompt_embeds') and self.sdxl_negative_pooled_prompt_embeds is not None:
                print(f"  Shape of self.sdxl_negative_pooled_prompt_embeds (from prepare): {self.sdxl_negative_pooled_prompt_embeds.shape}")
            else:
                print(f"  self.sdxl_negative_pooled_prompt_embeds: Not Found or None")
            if added_cond_kwargs is not None:
                print(f"  Shape of final text_embeds in added_cond_kwargs: {added_cond_kwargs.get('text_embeds').shape if added_cond_kwargs.get('text_embeds') is not None else 'Not in kwargs'}")
                print(f"  Shape of final time_ids in added_cond_kwargs: {added_cond_kwargs.get('time_ids').shape if added_cond_kwargs.get('time_ids') is not None else 'Not in kwargs'}")
            else:
                print("  added_cond_kwargs is None (e.g., not SDXL or error)")
            # ==== END SDXL DIAGNOSTIC LOGGING (StreamDiffusion.unet_step) ====
            added_cond_kwargs_xl_float32 = {
                "text_embeds": _current_text_embeds.to(dtype=torch.float32),
                "time_ids": _current_add_time_ids.to(dtype=torch.float32)
            }
        else:
            print(f"[StreamDiffusion.unet_step] Non-SDXL Mode or is_xl not set")
            added_cond_kwargs_xl_float32 = None

        print(f"[StreamDiffusion.unet_step] Shape of encoder_hidden_states (main prompt embeds) being passed to UNet: {self.prompt_embeds.shape}")
        print(f"[StreamDiffusion.unet_step] Shape of x_t_latent_plus_uc being passed to UNet: {x_t_latent_plus_uc.shape}")
        print(f"[StreamDiffusion.unet_step] Shape of t_list being passed to UNet: {t_list.shape}")

        if torch.isnan(x_t_latent_plus_uc).any() or torch.isinf(x_t_latent_plus_uc).any():
            print(f"[StreamDiffusion.unet_step] WARNING: UNet input latents (x_t_latent_plus_uc) contains NaN/inf values BEFORE UNet call!")

        # Store original UNet dtype and cast to float32 for stable computation
        original_unet_dtype = self.unet.dtype
        self.unet.to(torch.float32)

        model_pred = self.unet(
            x_t_latent_plus_uc.to(dtype=torch.float32),
            t_list.to(dtype=torch.float32), # t_list is usually int/long, but ensure consistency if it were float
            encoder_hidden_states=self.prompt_embeds.to(dtype=torch.float32),
            added_cond_kwargs=added_cond_kwargs_xl_float32, # Use float32 version
            return_dict=False,
        )[0]

        # Cast UNet output back to original dtype and restore UNet dtype
        model_pred = model_pred.to(original_unet_dtype)
        self.unet.to(original_unet_dtype) # Restore UNet to original dtype

        # NaN/inf check for UNet output
        if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
            print(f"[StreamDiffusion.unet_step] WARNING: UNet output contains NaN/inf values!")
            print(f"[StreamDiffusion.unet_step] UNet output min: {model_pred.min().item() if model_pred.numel() > 0 else 'N/A'}, max: {model_pred.max().item() if model_pred.numel() > 0 else 'N/A'}, mean: {model_pred.mean().item() if model_pred.numel() > 0 else 'N/A'}")

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )  # ここコメントアウトでself out cfg
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (
            self.cfg_type == "self" or self.cfg_type == "initialize"
        ):
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [
                        self.alpha_prod_t_sqrt[1:],
                        torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        self.beta_prod_t_sqrt[1:],
                        torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat(
                    [self.init_noise[1:], self.init_noise[0:1]], dim=0
                )
                self.stock_noise = init_noise + delta_x

        else:
            # denoised_batch = self.scheduler.step(model_pred, t_list[0], x_t_latent).denoised
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        if torch.isnan(image_tensors).any() or torch.isinf(image_tensors).any():
            print(f"[StreamDiffusion.encode_image] WARNING: Input image_tensors contains NaN/inf values!")
            print(f"[StreamDiffusion.encode_image] image_tensors min: {image_tensors.min().item() if image_tensors.numel() > 0 else 'N/A'}, max: {image_tensors.max().item() if image_tensors.numel() > 0 else 'N/A'}, mean: {image_tensors.mean().item() if image_tensors.numel() > 0 else 'N/A'}")

        image_tensors_for_vae = image_tensors.to(
            device=self.device,
            dtype=torch.float32, 
        )

        # VAE Input Logging Block
        print(f"[StreamDiffusion.encode_image] VAE Input (image_tensors_for_vae) stats before encode:")
        print(f"  - dtype: {image_tensors_for_vae.dtype}")
        print(f"  - device: {image_tensors_for_vae.device}")
        print(f"  - shape: {image_tensors_for_vae.shape}")
        if torch.isnan(image_tensors_for_vae).any() or torch.isinf(image_tensors_for_vae).any():
            print(f"  - WARNING: Contains NaN/inf values BEFORE VAE encode!")
        else:
            print(f"  - Does NOT contain NaN/inf values BEFORE VAE encode.")
        print(f"  - min: {image_tensors_for_vae.min().item() if image_tensors_for_vae.numel() > 0 else 'N/A'}")
        print(f"  - max: {image_tensors_for_vae.max().item() if image_tensors_for_vae.numel() > 0 else 'N/A'}")
        print(f"  - mean: {image_tensors_for_vae.mean().item() if image_tensors_for_vae.numel() > 0 else 'N/A'}")
        
        original_vae_dtype = self.vae.dtype
        try:
            if hasattr(self, 'is_xl') and self.is_xl and self.vae.dtype != torch.float32:
                print(f"[StreamDiffusion.encode_image] Temporarily casting VAE to float32 for encode. Original dtype: {original_vae_dtype}")
                self.vae = self.vae.to(torch.float32)
            
            encoded_output = self.vae.encode(image_tensors_for_vae)
        finally:
            if hasattr(self, 'is_xl') and self.is_xl and self.vae.dtype != original_vae_dtype: # Check if cast happened
                print(f"[StreamDiffusion.encode_image] Casting VAE back to original dtype: {original_vae_dtype}")
                self.vae = self.vae.to(original_vae_dtype)
        # Check the sample from the latent distribution
        # Sample once and reuse for checking and for retrieve_latents if it expects a sample
        # However, retrieve_latents might do its own sampling or use .mean. Assuming it handles encoded_output.
        sampled_latents_for_check = encoded_output.latent_dist.sample(generator=self.generator)
        
        if torch.isnan(sampled_latents_for_check).any() or torch.isinf(sampled_latents_for_check).any():
            print(f"[StreamDiffusion.encode_image] WARNING: VAE encoded_output.latent_dist.sample() contains NaN/inf values!")
            mean_val = encoded_output.latent_dist.mean
            std_val = encoded_output.latent_dist.std
            print(f"[StreamDiffusion.encode_image] VAE encoded_output mean min: {mean_val.min().item() if mean_val.numel() > 0 else 'N/A'}, max: {mean_val.max().item() if mean_val.numel() > 0 else 'N/A'}")
            print(f"[StreamDiffusion.encode_image] VAE encoded_output std min: {std_val.min().item() if std_val.numel() > 0 else 'N/A'}, max: {std_val.max().item() if std_val.numel() > 0 else 'N/A'}")

        img_latent = retrieve_latents(encoded_output, self.generator)
        
        if torch.isnan(img_latent).any() or torch.isinf(img_latent).any():
            print(f"[StreamDiffusion.encode_image] WARNING: img_latent (after retrieve_latents) contains NaN/inf values!")
            print(f"[StreamDiffusion.encode_image] img_latent min: {img_latent.min().item() if img_latent.numel() > 0 else 'N/A'}, max: {img_latent.max().item() if img_latent.numel() > 0 else 'N/A'}, mean: {img_latent.mean().item() if img_latent.numel() > 0 else 'N/A'}")

        img_latent_scaled = img_latent * self.vae.config.scaling_factor
        if torch.isnan(img_latent_scaled).any() or torch.isinf(img_latent_scaled).any():
            print(f"[StreamDiffusion.encode_image] WARNING: img_latent (after scaling_factor) contains NaN/inf values!")
            print(f"[StreamDiffusion.encode_image] img_latent (scaled) min: {img_latent_scaled.min().item() if img_latent_scaled.numel() > 0 else 'N/A'}, max: {img_latent_scaled.max().item() if img_latent_scaled.numel() > 0 else 'N/A'}, mean: {img_latent_scaled.mean().item() if img_latent_scaled.numel() > 0 else 'N/A'}")

        if self.init_noise is not None and (torch.isnan(self.init_noise[0]).any() or torch.isinf(self.init_noise[0]).any()):
            print(f"[StreamDiffusion.encode_image] WARNING: self.init_noise[0] contains NaN/inf values before add_noise!")

        img_latent_for_add_noise = img_latent_scaled.to(device=self.device, dtype=self.dtype)

        x_t_latent = self.add_noise(img_latent_for_add_noise, self.init_noise[0], 0) 
        if torch.isnan(x_t_latent).any() or torch.isinf(x_t_latent).any():
            print(f"[StreamDiffusion.encode_image] WARNING: x_t_latent (output of encode_image) contains NaN/inf values!")
            print(f"[StreamDiffusion.encode_image] x_t_latent min: {x_t_latent.min().item() if x_t_latent.numel() > 0 else 'N/A'}, max: {x_t_latent.max().item() if x_t_latent.numel() > 0 else 'N/A'}, mean: {x_t_latent.mean().item() if x_t_latent.numel() > 0 else 'N/A'}")
        
        return x_t_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        # NaN/inf check for VAE input (x_0_pred_out)
        if torch.isnan(x_0_pred_out).any() or torch.isinf(x_0_pred_out).any():
            print(f"[StreamDiffusion.decode_image] WARNING: VAE input (x_0_pred_out) contains NaN/inf values!")
            print(f"[StreamDiffusion.decode_image] VAE input min: {x_0_pred_out.min().item() if x_0_pred_out.numel() > 0 else 'N/A'}, max: {x_0_pred_out.max().item() if x_0_pred_out.numel() > 0 else 'N/A'}, mean: {x_0_pred_out.mean().item() if x_0_pred_out.numel() > 0 else 'N/A'}")

        image_latents = x_0_pred_out / self.vae.config.scaling_factor

        # NaN/inf check for VAE input scaled (before float32 cast)
        if torch.isnan(image_latents).any() or torch.isinf(image_latents).any():
            print(f"[StreamDiffusion.decode_image] WARNING: Scaled VAE input (image_latents) BEFORE float32 cast contains NaN/inf values!")
            print(f"[StreamDiffusion.decode_image] image_latents (before cast) min: {image_latents.min().item() if image_latents.numel() > 0 else 'N/A'}, max: {image_latents.max().item() if image_latents.numel() > 0 else 'N/A'}, mean: {image_latents.mean().item() if image_latents.numel() > 0 else 'N/A'}")

        # Ensure VAE decode runs in float32 for stability
        image_latents_for_decode = image_latents.to(dtype=torch.float32)
        print(f"[StreamDiffusion.decode_image] VAE decode input (image_latents_for_decode) dtype: {image_latents_for_decode.dtype}, device: {image_latents_for_decode.device}")
        
        # NaN/inf check for VAE input after float32 cast
        if torch.isnan(image_latents_for_decode).any() or torch.isinf(image_latents_for_decode).any():
            print(f"[StreamDiffusion.decode_image] WARNING: image_latents_for_decode (AFTER float32 cast) still contains NaN/inf values!")
            print(f"[StreamDiffusion.decode_image] image_latents_for_decode (after cast) min: {image_latents_for_decode.min().item() if image_latents_for_decode.numel() > 0 else 'N/A'}, max: {image_latents_for_decode.max().item() if image_latents_for_decode.numel() > 0 else 'N/A'}, mean: {image_latents_for_decode.mean().item() if image_latents_for_decode.numel() > 0 else 'N/A'}")

        original_vae_dtype = self.vae.dtype
        try:
            if hasattr(self, 'is_xl') and self.is_xl and self.vae.dtype != torch.float32:
                print(f"[StreamDiffusion.decode_image] Temporarily casting VAE to float32 for decode. Original dtype: {original_vae_dtype}")
                self.vae = self.vae.to(torch.float32)

            image = self.vae.decode(
                image_latents_for_decode, return_dict=False
            )[0]
        finally:
            if hasattr(self, 'is_xl') and self.is_xl and self.vae.dtype != original_vae_dtype: # Check if cast happened
                print(f"[StreamDiffusion.decode_image] Casting VAE back to original dtype: {original_vae_dtype}")
                self.vae = self.vae.to(original_vae_dtype)

        # NaN/inf check for VAE output (before normalization)
        if torch.isnan(image).any() or torch.isinf(image).any():
            print(f"[StreamDiffusion.decode_image] WARNING: VAE output (before normalization) contains NaN/inf values!")
            print(f"[StreamDiffusion.decode_image] VAE output (raw) min: {image.min().item() if image.numel() > 0 else 'N/A'}, max: {image.max().item() if image.numel() > 0 else 'N/A'}, mean: {image.mean().item() if image.numel() > 0 else 'N/A'}")

        return image

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        prev_latent_batch = self.x_t_latent_buffer

        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                self.stock_noise = torch.cat(
                    (self.init_noise[0:1], self.stock_noise[:-1]), dim=0
                )
            x_0_pred_batch, model_pred = self.unet_step(x_t_latent, t_list)

            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.do_add_noise:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                    )
                else:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    )
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )

                # SDXL conditioning for this non-batch path
                current_prompt_embeds = self.prompt_embeds
                if self.is_xl:
                    bs = x_t_latent.shape[0] # x_t_latent is [N, C, H, W]
                    _current_pooled_embeds = self.sdxl_pooled_prompt_embeds.repeat(bs, 1)
                    _current_add_time_ids = self.sdxl_add_time_ids.repeat(bs, 1)

                    # Ensure dtypes are consistent for added_cond_kwargs
                    _current_pooled_embeds = _current_pooled_embeds.to(dtype=x_t_latent.dtype, device=self.device)
                    _current_add_time_ids = _current_add_time_ids.to(dtype=x_t_latent.dtype, device=self.device)
                    added_cond_kwargs_xl = {"text_embeds": _current_pooled_embeds, "time_ids": _current_add_time_ids}
                else:
                    added_cond_kwargs_xl = None
                
                current_prompt_embeds = current_prompt_embeds.to(dtype=x_t_latent.dtype, device=self.device)

                # Store original UNet dtype and cast UNet to float32 for computation
                original_unet_dtype = self.unet.dtype
                try:
                    self.unet = self.unet.to(torch.float32)

                    # Use t_reshaped (which is `t` from loop, reshaped and on device)
                    latent_model_input = self.scheduler.scale_model_input(x_t_latent, t_reshaped)
                    latent_model_input_fp32 = latent_model_input.to(torch.float32)
                    
                    current_prompt_embeds_fp32 = current_prompt_embeds.to(torch.float32)
                    added_cond_kwargs_xl_fp32 = None
                    if added_cond_kwargs_xl:
                        added_cond_kwargs_xl_fp32 = {
                            "text_embeds": added_cond_kwargs_xl["text_embeds"].to(torch.float32),
                            "time_ids": added_cond_kwargs_xl["time_ids"].to(torch.float32),
                        }

                    # Direct UNet call
                    noise_pred_fp32 = self.unet(
                        latent_model_input_fp32,
                        t_reshaped.to(torch.float32),
                        encoder_hidden_states=current_prompt_embeds_fp32,
                        added_cond_kwargs=added_cond_kwargs_xl_fp32,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_fp32.to(original_unet_dtype) # Cast result back
                finally:
                    self.unet = self.unet.to(original_unet_dtype) # Restore UNet dtype

                # Scheduler step
                x_t_latent = self.scheduler.step(noise_pred, t_reshaped, x_t_latent).prev_sample
            
            # After the loop, the final latent is the output for this path
            x_0_pred_out = x_t_latent
        


        return x_0_pred_out

    @torch.no_grad()
    def __call__(
        self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None
    ) -> torch.Tensor:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if x is not None:
            x = self.image_processor.preprocess(x, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )
            if self.similar_image_filter:
                x = self.similar_filter(x)
                if x is None:
                    time.sleep(self.inference_time_ema)
                    return self.prev_image_result
            x_t_latent = self.encode_image(x)
        else:
            # TODO: check the dimension of x_t_latent
            x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        x_output = self.decode_image(x_0_pred_out).detach().clone()

        self.prev_image_result = x_output
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        return x_output

    @torch.no_grad()
    def txt2img(self, batch_size: int = 1) -> torch.Tensor:
        x_0_pred_out = self.predict_x0_batch(
            torch.randn((batch_size, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        )
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        return x_output

    def txt2img_sd_turbo(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = torch.randn(
            (batch_size, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype,
        )
        model_pred = self.unet(
            x_t_latent,
            self.sub_timesteps_tensor,
            encoder_hidden_states=self.prompt_embeds,
            return_dict=False,
        )[0]
        x_0_pred_out = (
            x_t_latent - self.beta_prod_t_sqrt * model_pred
        ) / self.alpha_prod_t_sqrt
        return self.decode_image(x_0_pred_out)
