import gc
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
import tqdm

import saver
from cross_attention_collector import trace
from tadisr_pipelines import TADiSRPipeline
from dataset import parse_args_testing, ResizeLongestSide, FTSRRealTestDataset, pad_to_max_length


def extract_overlapping_patches(img, patch_size, overlap):
    """
    img: Tensor of shape (B, C, H_up, W_up)
    patch_size: tuple (patch_H, patch_W)
    overlap: tuple (overlap_H, overlap_W)
    """
    patch_H, patch_W = patch_size
    overlap_H, overlap_W = overlap

    stride_H = patch_H - overlap_H
    stride_W = patch_W - overlap_W

    # Use unfold to extract sliding patches
    patches = img.unfold(2, patch_H, stride_H).unfold(3, patch_W, stride_W)

    # Rearrange to (B * num_patches, C, patch_H, patch_W)
    B, C, nH, nW, pH, pW = patches.shape
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B * nH * nW, C, pH, pW)

    return patches


def combine_patches(patches, out_shape, patch_size, overlap):
    """
    patches: Tensor of shape (N, C, H_patch, W_patch)
    out_shape: tuple (B, C, H_out, W_out)
    patch_size: tuple (H_patch, W_patch)
    overlap: tuple (overlap_H, overlap_W)
    """
    B, C, H_out, W_out = out_shape
    H_patch, W_patch = patch_size
    stride_H = H_patch - overlap[0]
    stride_W = W_patch - overlap[1]

    output = torch.zeros((B, C, H_out, W_out), device=patches.device)
    count_map = torch.zeros((B, 1, H_out, W_out), device=patches.device)

    idx = 0
    nH = (H_out - overlap[0]) // stride_H
    nW = (W_out - overlap[1]) // stride_W

    for i in range(nH):
        for j in range(nW):
            top = i * stride_H
            left = j * stride_W
            bottom = top + H_patch
            right = left + W_patch

            output[:, :, top:bottom, left:right] += patches[idx]
            count_map[:, :, top:bottom, left:right] += 1
            idx += 1

    # 避免除以0
    count_map = torch.clamp(count_map, min=1.0)
    output = output / count_map

    return output


def combine_sr_patches_weighted(patches, out_shape, patch_size, overlap):
    B, C, H_out, W_out = out_shape
    H_patch, W_patch = patch_size
    stride_H = H_patch - overlap[0]
    stride_W = W_patch - overlap[1]

    output = torch.zeros((B, C, H_out, W_out), device=patches.device)
    weight_map = torch.zeros((B, 1, H_out, W_out), device=patches.device)

    h_weight = torch.linspace(0, 1, steps=H_patch, device=patches.device).unsqueeze(1).expand(H_patch, W_patch)
    w_weight = torch.linspace(0, 1, steps=W_patch, device=patches.device).unsqueeze(0).expand(H_patch, W_patch)
    weight = torch.minimum(h_weight, 1 - h_weight) * torch.minimum(w_weight, 1 - w_weight)  # 形成中间强、边缘弱的2D mask
    weight = weight.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H_patch, W_patch)

    idx = 0
    nH = (H_out - overlap[0]) // stride_H
    nW = (W_out - overlap[1]) // stride_W

    for i in range(nH):
        for j in range(nW):
            top = i * stride_H
            left = j * stride_W
            bottom = top + H_patch
            right = left + W_patch

            patch = patches[idx].unsqueeze(0)
            w = weight.expand_as(patch)

            output[:, :, top:bottom, left:right] += patch * w
            weight_map[:, :, top:bottom, left:right] += w[:1, :1, :, :]  # 只加一次通道权重
            idx += 1

    output = output / torch.clamp(weight_map, min=1e-6)
    return output


def combine_mask_patches(patches, out_shape, patch_size, overlap):
    B, C, H_out, W_out = out_shape
    H_patch, W_patch = patch_size
    stride_H = H_patch - overlap[0]
    stride_W = W_patch - overlap[1]

    output = torch.zeros((B, C, H_out, W_out), dtype=patches.dtype, device=patches.device)

    idx = 0
    nH = (H_out - overlap[0]) // stride_H
    nW = (W_out - overlap[1]) // stride_W

    for i in range(nH):
        for j in range(nW):
            top = i * stride_H
            left = j * stride_W
            bottom = top + H_patch
            right = left + W_patch

            output[:, :, top:bottom, left:right] = torch.maximum(
                output[:, :, top:bottom, left:right],
                patches[idx]
            )
            idx += 1

    return output


def main(args):
    device = torch.device("cuda")

    net_pix2pix = TADiSRPipeline()
    net_pix2pix.load_model("[weights]")
    net_pix2pix.set_eval()

    dataset_val = FTSRRealTestDataset(args.test_folder)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    net_pix2pix.to(device)
    scaled_factor = 4
    patch_size = 768
    for step, batch_val in enumerate(dl_val):
        if args.selected and batch_val['name'][0].split('.')[0] not in args.selected:
            continue

        lr_image = batch_val["input"]

        B, C, H, W = lr_image.shape
        assert B == 1, "Use batch size 1 for eval."
        patchH, patchW = patch_size, patch_size
        upH, upW = ResizeLongestSide.get_new_size((H, W), patch_size)
        lr_image_up = F.interpolate(lr_image, size=(upH * scaled_factor, upW * scaled_factor), mode='bilinear')
        lr_image_up = pad_to_max_length(lr_image_up, patch_size * scaled_factor)
        lr_image_tiled = extract_overlapping_patches(
            lr_image_up, patch_size=(patchH, patchW),
            overlap=(patchH // 2, patchW // 2)
        )
        sr_patches = []
        mask_patches = []
        with torch.no_grad():
            for patch in tqdm.tqdm(lr_image_tiled):
                with trace(net_pix2pix, context_size=256, locate_middle=False) as tc:
                    patch = patch.unsqueeze(0).cuda()
                    sr_patch, mask_patch = net_pix2pix(patch, tc=tc)
                    sr_patches.append(sr_patch.detach().cpu())
                    mask_patches.append(mask_patch.detach().cpu())

        sr_patches = torch.cat(sr_patches, dim=0)
        mask_patches = torch.cat(mask_patches, dim=0)

        # 拼接
        sr_output = combine_sr_patches_weighted(
            sr_patches, out_shape=(B, C, patchH * scaled_factor, patchW * scaled_factor),
            patch_size=(patchH, patchW), overlap=(patchH // 2, patchW // 2))
        mask_output = combine_mask_patches(
            mask_patches, out_shape=(B, 1, patchH * scaled_factor, patchW * scaled_factor),
            patch_size=(patchH, patchW), overlap=(patchH // 2, patchW // 2))

        saver.base_url = os.path.join(args.output_dir, "results", batch_val["name"][0])

        saver.save_image(lr_image, "lr_image")
        saver.save_image(sr_output, "sr_image")
        saver.save_image(mask_output, "mask")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args_testing()
    main(args)
