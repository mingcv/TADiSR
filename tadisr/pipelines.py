import os
from collections import OrderedDict
from typing import Optional, Union, List, Tuple, Dict, Any

import PIL
import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, FlowMatchEulerDiscreteScheduler, \
    CogView4Transformer2DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.activations import get_activation
from peft import LoraConfig
from transformers import AutoTokenizer, GlmModel
from tadisr.chatglm_tokenizer import ChatGLMTokenizer

try:
    from diffusers.models.unet_2d_blocks import UpDecoderBlock2D
except:
    from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CABlock(nn.Module):
    def __init__(self, channels):
        super(CABlock, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return x * self.ca(x)


class DualStreamGLU(nn.Module):
    def __init__(self):
        super(DualStreamGLU, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)

        return x1 * self.sigmoid(y2), y1 * self.sigmoid(x2)


class DualStreamSeq(nn.Sequential):
    def forward(self, x, y=None):
        y = y if y is not None else x
        for module in self:
            x, y = module(x, y)
        return x, y


class DualStreamBlock(nn.Module):
    def __init__(self, *args):
        super(DualStreamBlock, self).__init__()
        self.seq_l = nn.Sequential()
        self.seq_r = nn.Sequential()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.seq_l.add_module(key, module)
                self.seq_r.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.seq_l.add_module(str(idx), module)
                self.seq_r.add_module(str(idx), module)

    def forward(self, x, y):
        return self.seq_l(x), self.seq_r(y)


class ResnetBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            conv_shortcut: bool = False,
            dropout: float = 0.0,
            groups: int = 32,
            groups_out: Optional[int] = None,
            eps: float = 1e-6,
            non_linearity: str = "swish",
            skip_time_act: bool = False,
            use_in_shortcut: Optional[bool] = None,
            conv_shortcut_bias: bool = True,
            conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states
        return output_tensor


class CDIBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                ResnetBlock2D(c, c, non_linearity="silu"),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGLU(),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.GroupNorm(num_groups=32, num_channels=c)),
            DualStreamBlock(nn.SiLU()),
            DualStreamBlock(nn.Conv2d(c, c, 1)),
        )

        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp_l, inp_r):
        x, y = self.block1(inp_l, inp_r)
        out_l, out_r = inp_l + x * self.a_l, inp_r + y * self.a_r
        return out_l, out_r


def retrieve_latents(
        encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


class JointSegmentationDecoders(nn.Module):
    def __init__(self):
        super().__init__()
        # self.channel_mapping = nn.Conv2d(550, 512, 1)
        self.interaction_blocks = nn.ModuleList([
            DualStreamSeq(CDIBlock(1024)),
            DualStreamSeq(CDIBlock(1024)),
            DualStreamSeq(CDIBlock(1024)),
            DualStreamSeq(CDIBlock(512))
        ])
        self.up_blocks = nn.ModuleList([
            UpDecoderBlock2D(
                num_layers=2,
                in_channels=1024,
                out_channels=1024,
                add_upsample=True,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                resnet_groups=32,
                resnet_time_scale_shift="group"
            ),
            UpDecoderBlock2D(
                num_layers=2,
                in_channels=1024,
                out_channels=1024,
                add_upsample=True,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                resnet_groups=32,
                resnet_time_scale_shift="group"
            ),
            UpDecoderBlock2D(
                num_layers=2,
                in_channels=1024,
                out_channels=512,
                add_upsample=True,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                resnet_groups=32,
                resnet_time_scale_shift="group"
            ), UpDecoderBlock2D(
                num_layers=2,
                in_channels=512,
                out_channels=128,
                add_upsample=False,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                resnet_groups=32,
                resnet_time_scale_shift="group"
            )
        ])

        self.out_block = nn.Sequential(
            nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(128, 1, 3, padding=1)
        )


class JointSegmentationDecodersMuGI(nn.Module):
    def __init__(self):
        super().__init__()
        # self.channel_mapping = nn.Conv2d(550, 512, 1)
        self.interaction_blocks = nn.ModuleList([
            DualStreamSeq(CDIBlock(1024)),
            DualStreamSeq(CDIBlock(1024)),
            DualStreamSeq(CDIBlock(1024)),
            DualStreamSeq(CDIBlock(512))
        ])
        self.up_blocks = nn.ModuleList([
            UpDecoderBlock2D(
                num_layers=2,
                in_channels=1024,
                out_channels=1024,
                add_upsample=True,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                resnet_groups=32,
                resnet_time_scale_shift="group"
            ),
            UpDecoderBlock2D(
                num_layers=2,
                in_channels=1024,
                out_channels=1024,
                add_upsample=True,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                resnet_groups=32,
                resnet_time_scale_shift="group"
            ),
            UpDecoderBlock2D(
                num_layers=2,
                in_channels=1024,
                out_channels=512,
                add_upsample=True,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                resnet_groups=32,
                resnet_time_scale_shift="group"
            ), UpDecoderBlock2D(
                num_layers=2,
                in_channels=512,
                out_channels=128,
                add_upsample=False,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                resnet_groups=32,
                resnet_time_scale_shift="group"
            )
        ])

        self.out_block = nn.Sequential(
            nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(128, 1, 3, padding=1)
        )


class TADiSRPipeline(nn.Module):
    def __init__(self):
        super().__init__()

        ckpt_dir = "./weights/Kolors"
        vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)

        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False

        torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
        target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                              "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                              "to_k", "to_q", "to_v", "to_out.0",
                              ]
        vae_lora_config = LoraConfig(r=2, init_lora_weights="gaussian",
                                     target_modules=target_modules_vae)
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae = vae
        self.vae.to("cuda")

        self.unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet")
        self.target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]
        unet_lora_config = LoraConfig(r=4, init_lora_weights="gaussian", target_modules=self.target_modules_unet)
        self.unet.add_adapter(unet_lora_config)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.unet.config.sample_size

        prompt_path = "[pre-computed prompt tokens]"
        saved_prompt_tokens = torch.load(prompt_path, map_location="cuda")
        self.prompt_embeds = saved_prompt_tokens["prompt_embeds"].float().cuda()
        self.added_cond_kwargs = {
            "text_embeds": saved_prompt_tokens["added_cond_kwargs"]["text_embeds"].float().cuda(),
            "time_ids": saved_prompt_tokens["added_cond_kwargs"]["time_ids"].float().cuda()
        }

        self.timesteps = torch.tensor([200], device="cuda").long()
        self.scheduler = make_1step_sched()
        self.tokenizer = ChatGLMTokenizer.from_pretrained(f"{ckpt_dir}/tokenizer")
        self.js_decoder = JointSegmentationDecoders()

        self.execution_device = torch.device("cuda")
        self.to_device()

    def to_device(self):
        self.unet.to(self.execution_device)
        self.vae.to(self.execution_device)
        self.js_decoder.to(self.execution_device)

    def set_eval(self):
        self.unet.eval()
        self.unet.requires_grad_(False)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.js_decoder.eval()

    def set_train(self):
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

        self.vae.train()
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

        self.js_decoder.train()

    def save_model(self, outf):
        sd = {}
        sd["mi_decoder"] = self.js_decoder.state_dict()
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["rank_unet"] = 4
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)

    def load_model_pretrain(self, state_dict_path):
        state_dict = torch.load(state_dict_path)
        ret = self.unet.load_state_dict(state_dict["state_dict_unet"], strict=False)
        print("unet load lora:", ret)

    def load_model(self, state_dict_path):
        state_dict = torch.load(state_dict_path)
        ret = self.unet.load_state_dict(state_dict["state_dict_unet"], strict=False)
        ret = self.vae.load_state_dict(state_dict["state_dict_vae"], strict=False)
        decoder_key = "js_decoder" if "js_decoder" in state_dict else "mi_decoder"
        ret = self.js_decoder.load_state_dict(state_dict[decoder_key], strict=True)
        print("js_decoder: ", ret)

    def _get_add_time_ids(
            self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def get_guidance_scale_embedding(
            self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def prepare_latents(
            self, image, batch_size, num_images_per_prompt, dtype, device, generator=None
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        latents_mean = latents_std = None
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None:
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1)

        # Offload text encoder if `enable_model_cpu_offload` was enabled
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            # make sure the VAE is in float32 mode, as it overflows in float16
            if self.vae.config.force_upcast:
                image = image.float()
                self.vae.to(dtype=torch.float32)

            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                    image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
                elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                    )

                init_latents = [
                    retrieve_latents(self.vae.encode(image[i: i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            if self.vae.config.force_upcast:
                self.vae.to(dtype)

            init_latents = init_latents.to(dtype)
            if latents_mean is not None and latents_std is not None:
                latents_mean = latents_mean.to(device=device, dtype=dtype)
                latents_std = latents_std.to(device=device, dtype=dtype)
                init_latents = (init_latents - latents_mean) * self.vae.config.scaling_factor / latents_std
            else:
                init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        latents = init_latents

        return latents

    def forward(self, image=None,
                guidance_scale: float = 5.0,
                num_images_per_prompt: Optional[int] = 1,
                tc=None):
        image = self.image_processor.preprocess(image)

        self._guidance_scale = guidance_scale

        # 2. Define call parameters
        batch_size = 1
        device = self.execution_device
        # 5. Prepare latent variables
        latents = self.prepare_latents(
            image,
            batch_size,
            num_images_per_prompt,
            self.prompt_embeds.dtype,
            device,
        )

        prompt_embeds = self.prompt_embeds.to(device)

        # 8. Denoising loop
        latent_model_input = latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, self.timesteps)

        # predict the noise residual

        noise_pred = self.unet(
            latent_model_input,
            self.timesteps,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            added_cond_kwargs=self.added_cond_kwargs,
            return_dict=False,
        )[0]

        latents = self.scheduler.step(noise_pred, self.timesteps, latents, return_dict=False)[0]
        heat_map = tc.compute_global_heat_map("best quality,highres,extremely detailed,text")
        attn_map = heat_map.compute_word_heat_map_kolors('text').heatmap

        latents_out = latents / self.vae.config.scaling_factor
        latents_out = self.vae.post_quant_conv(latents_out)
        latents_out = self.vae.decoder.conv_in(latents_out)
        latents_out = self.vae.decoder.mid_block(latents_out)

        attn_map = self.js_decoder.channel_mapping(attn_map)
        skip_convs = [self.vae.decoder.skip_conv_1, self.vae.decoder.skip_conv_2,
                      self.vae.decoder.skip_conv_3, self.vae.decoder.skip_conv_4]
        for i, (up_block_im, up_block_mask,
                inter_block) in enumerate(zip(self.vae.decoder.up_blocks,
                                              self.js_decoder.up_blocks,
                                              self.js_decoder.interaction_blocks)):
            if i > 0:
                skip_in = skip_convs[i](self.vae.encoder.current_down_blocks[::-1][i])
                latents_out = latents_out + skip_in

            latents_out, attn_map = inter_block(latents_out, attn_map)
            latents_out = up_block_im(latents_out)
            attn_map = up_block_mask(attn_map)

        mask_out = self.js_decoder.out_block(attn_map)
        latents_out = self.vae.decoder.conv_norm_out(latents_out)
        latents_out = self.vae.decoder.conv_act(latents_out)
        sr_image = self.vae.decoder.conv_out(latents_out)
        sr_image = self.image_processor.postprocess(sr_image, output_type="pt")
        return sr_image, mask_out


class CogView4TextEncoderPipeline(nn.Module):
    def __init__(self, ckpt_dir="./weights/CogView4", device="cuda"):
        super().__init__()
        self.execution_device = torch.device(device)
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"CogView4 base model directory not found: {ckpt_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(f"{ckpt_dir}/tokenizer")
        self.text_encoder = GlmModel.from_pretrained(f"{ckpt_dir}/text_encoder").to(self.execution_device)

    def _get_glm_embeds(
            self,
            prompt: Union[str, List[str]] = None,
            max_sequence_length: int = 1024,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        dtype = self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer(
            prompt,
            padding="longest",  # not use max length
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        current_length = text_input_ids.shape[1]
        pad_length = (16 - (current_length % 16)) % 16
        if pad_length > 0:
            pad_ids = torch.full(
                (text_input_ids.shape[0], pad_length),
                fill_value=self.tokenizer.pad_token_id,
                dtype=text_input_ids.dtype,
                device=text_input_ids.device,
            )
            text_input_ids = torch.cat([pad_ids, text_input_ids], dim=1)
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=True).hidden_states[-2]

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def encode_prompt(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: Optional[Union[str, List[str]]] = None,
            do_classifier_free_guidance: bool = True,
            num_images_per_prompt: int = 1,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            max_sequence_length: int = 1024,
    ):

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_glm_embeds(prompt, max_sequence_length, device, dtype)

        seq_len = prompt_embeds.size(1)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_glm_embeds(negative_prompt, max_sequence_length, device, dtype)

            seq_len = negative_prompt_embeds.size(1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def prepare_text_embed(self, prompt, negative_prompt, save_path):
        device = self.execution_device

        # Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance=False,
            num_images_per_prompt=1,
            device=device,
        )
        torch.save({
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds
        }, save_path)

        print(f"Saved prompt embeds in {save_path}")


class CogView4Pipeline(nn.Module):
    """TADiSR inference model built on a local CogView4-6B Diffusers export.

    The checkpoint contains only LoRA adapters and the joint segmentation decoder;
    the public CogView4 base model is supplied separately through ``ckpt_dir``.
    """

    def __init__(self, ckpt_dir="./weights/CogView4", prompt_path=None, device="cuda"):
        super().__init__()
        self.execution_device = torch.device(device)
        if self.execution_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
        if prompt_path is None:
            prompt_path = f"{ckpt_dir}/saved_prompt_tokens_nocfg.pt"
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"CogView4 base model directory not found: {ckpt_dir}")
        if not os.path.isfile(prompt_path):
            raise FileNotFoundError(
                f"Prompt embeddings not found: {prompt_path}. Run prepare_cogview4_prompt_embeddings.py first."
            )
        vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)

        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.ignore_skip = False

        torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
        target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                              "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                              "to_k", "to_q", "to_v", "to_out.0",
                              ]
        vae_lora_config = LoraConfig(r=2, init_lora_weights="gaussian",
                                     target_modules=target_modules_vae)
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae = vae
        self.vae.to(self.execution_device)

        self.transformer = CogView4Transformer2DModel.from_pretrained(f"{ckpt_dir}/transformer")
        self.target_modules_transformer = [
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
            "ff.net.0.proj", "ff.net.2",
            "proj_in", "proj_out"
        ]
        transformer_lora_config = LoraConfig(
            r=4, init_lora_weights="gaussian",
            target_modules=self.target_modules_transformer
        )
        self.transformer.add_adapter(transformer_lora_config)
        self.transformer.to(self.execution_device)

        self.js_decoder = JointSegmentationDecoders()
        self.js_decoder.to(self.execution_device)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
        self.target_timestep = 200.0

        t_diff = torch.abs(self.scheduler.timesteps - self.target_timestep)
        self.closest_idx = torch.argmin(t_diff)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        saved_prompt_tokens = torch.load(prompt_path, map_location="cpu")
        self.prompt_embeds = saved_prompt_tokens["prompt_embeds"].float().to(self.execution_device)

    def set_eval(self):
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.js_decoder.eval()

    def set_train(self):
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.transformer.train()
        for n, _p in self.transformer.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

        self.vae.train()
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

        self.js_decoder.train()

    def save_model(self, outf):
        sd = {}
        sd["js_decoder"] = self.js_decoder.state_dict()
        sd["transformer_lora_target_modules"] = self.target_modules_transformer
        sd["rank_transformer"] = 4
        sd["state_dict_transformer"] = {k: v for k, v in self.transformer.state_dict().items() if "lora" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)

    def load_model(self, state_dict_path):
        state_dict = torch.load(state_dict_path, map_location="cpu")
        ret = self.transformer.load_state_dict(state_dict["state_dict_transformer"], strict=False)
        # print("transformer", ret)
        ret = self.vae.load_state_dict(state_dict["state_dict_vae"], strict=False)
        # print("vae", ret)
        ret = self.js_decoder.load_state_dict(state_dict["js_decoder"], strict=True)
        print("js_decoder: ", ret)

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=self.execution_device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device=self.execution_device, dtype=dtype)
        timesteps = timesteps.to(device=self.execution_device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(self, lr_image, num_images_per_prompt: int = 1):
        batch_size = lr_image.shape[0]
        height, width = lr_image.shape[-2:]

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        indices = torch.full((batch_size,), self.closest_idx.item(), dtype=torch.long)
        timesteps = self.scheduler.timesteps[indices]
        sigmas = self.get_sigmas(timesteps, n_dim=lr_image.ndim, dtype=lr_image.dtype)

        device = self.execution_device

        prompt_embeds = self.prompt_embeds.repeat(batch_size, 1, 1)
        lr_image = self.image_processor.preprocess(lr_image)

        vae_shift_factor = 0
        latents = self.vae.encode(lr_image).latent_dist.sample()
        latents = (latents - vae_shift_factor) * self.vae.config.scaling_factor

        # Prepare additional timestep conditions
        original_size = torch.tensor([original_size], dtype=prompt_embeds.dtype, device=device)
        target_size = torch.tensor([target_size], dtype=prompt_embeds.dtype, device=device)
        crops_coords_top_left = torch.tensor([crops_coords_top_left], dtype=prompt_embeds.dtype, device=device)

        original_size = original_size.repeat(batch_size * num_images_per_prompt, 1)
        target_size = target_size.repeat(batch_size * num_images_per_prompt, 1)
        crops_coords_top_left = crops_coords_top_left.repeat(batch_size * num_images_per_prompt, 1)

        latent_model_input = latents.to(self.transformer.dtype)
        timesteps = timesteps.to(device)

        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            original_size=original_size,
            target_size=target_size,
            crop_coords=crops_coords_top_left,
            return_dict=False,
        )[0]
        latents = latents - sigmas * noise_pred

        latents_out = latents / self.vae.config.scaling_factor
        latents_out = self.vae.decoder.conv_in(latents_out)
        latents_out = self.vae.decoder.mid_block(latents_out)

        attn_map = latents_out
        skip_convs = [self.vae.decoder.skip_conv_1, self.vae.decoder.skip_conv_2,
                      self.vae.decoder.skip_conv_3, self.vae.decoder.skip_conv_4]

        for i, (up_block_im, up_block_mask,
                inter_block) in enumerate(zip(self.vae.decoder.up_blocks,
                                              self.js_decoder.up_blocks,
                                              self.js_decoder.interaction_blocks)):

            if i > 0:
                skip_in = skip_convs[i](self.vae.encoder.current_down_blocks[::-1][i])
                latents_out = latents_out + skip_in

            latents_out, attn_map = inter_block(latents_out, attn_map)
            latents_out = up_block_im(latents_out)
            attn_map = up_block_mask(attn_map)

        mask_out = self.js_decoder.out_block(attn_map)
        latents_out = self.vae.decoder.conv_norm_out(latents_out)
        latents_out = self.vae.decoder.conv_act(latents_out)
        sr_image = self.vae.decoder.conv_out(latents_out)
        sr_image = self.image_processor.postprocess(sr_image, output_type="pt")

        return sr_image, mask_out
