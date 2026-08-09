import gc
import os

import diffusers
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
import transformers
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

import saver
from losses import get_edge_loss_bbox, DifferentiableEdgeDetector
from metrics import PSNR, SSIM
from ppocr.ppocr_model import TextSystem
from tadisr_pipelines import CogView4Pipeline
from utils.training_utils import parse_args_paired_training, FTSRRealCEDatasetScaled, \
    FTSRRealCEDataset512


def print_in_main_process(accelerator, text):
    if accelerator.is_main_process:
        print(text)


def mask_losses(
        inputs: torch.Tensor,
        targets: torch.Tensor):
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    return sigmoid_focal_loss(inputs, targets), dice_loss(inputs, targets)


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2,
):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks=1.0):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def focal_mse(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask_input: torch.Tensor,
        mask_target: torch.Tensor,
        gamma: float = 2
):
    prob = mask_input.sigmoid()
    p_t = prob * mask_target + (1 - prob) * (1 - mask_target)
    loss = F.mse_loss(inputs, targets, reduction="none") * ((1 - p_t) ** gamma)
    return loss.mean()


def custom_collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        if key == 'name' or key == 'mask_ori':
            batch_dict[key] = [item[key] for item in batch]
        else:
            batch_dict[key] = torch.stack([item[key] for item in batch])

    return batch_dict


def main(args):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
        kwargs_handlers=[kwargs]
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    if not args.pretrained_model_name_or_path or not args.prompt_embeddings:
        raise ValueError("--pretrained_model_name_or_path and --prompt_embeddings are required for CogView4 training.")
    if not args.realce_train_list or not args.realce_eval_list:
        raise ValueError("--realce_train_list and --realce_eval_list are required for Real-CE fine-tuning.")
    net_pix2pix = CogView4Pipeline(
        ckpt_dir=args.pretrained_model_name_or_path,
        prompt_path=args.prompt_embeddings,
        device=accelerator.device,
    )
    if args.resume_checkpoint:
        net_pix2pix.load_model(args.resume_checkpoint)
    net_pix2pix.set_train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pix2pix.transformer.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)

    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)

    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
                    list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
                    list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
                    list(net_pix2pix.vae.decoder.skip_conv_4.parameters())

    for n, _p in net_pix2pix.js_decoder.named_parameters():
        if _p.requires_grad:
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon, )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power, )

    dataset_train = FTSRRealCEDatasetScaled(args.train_folders[0], args.realce_train_list)

    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                           num_workers=args.dataloader_num_workers, collate_fn=custom_collate_fn)
    dataset_val = FTSRRealCEDataset512(args.test_folder, args.realce_eval_list)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    ocr_model = TextSystem()
    # Prepare everything with our `accelerator`.
    net_pix2pix, optimizer, dl_train, lr_scheduler, net_lpips = accelerator.prepare(
        net_pix2pix, optimizer, dl_train, lr_scheduler, net_lpips
    )

    weight_dtype = torch.float32

    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # turn off eff. attn for the discriminator
    print_in_main_process(accelerator, "=========== Start the training loop ===========")
    global_step = 0
    edge_detector = DifferentiableEdgeDetector().cuda()
    for epoch in range(0, args.num_training_epochs):
        progress_bar = tqdm(range(0, len(dl_train)), initial=0, desc="Steps",
                            disable=not accelerator.is_local_main_process)

        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix]
            with accelerator.accumulate(*l_acc):
                lr_image = batch['input']
                hr_image = batch['gt']
                mask_gt = batch['mask']
                mask_gt_ori = batch['mask_ori']
                boxes, rec_res = ocr_model(mask_gt_ori[0])
                # print(boxes, rec_res)
                # forward pass
                sr_image, mask_pred = net_pix2pix(lr_image)

                loss_l2 = F.mse_loss(sr_image, hr_image, reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(sr_image, hr_image).mean() * args.lambda_lpips
                loss_edge = get_edge_loss_bbox(edge_detector, sr_image, hr_image, [boxes])
                loss_edge = loss_edge * 5.0

                loss_mask = F.mse_loss(mask_pred, mask_gt, reduction="mean") * args.lambda_l2
                loss_mask_dice, loss_mask_focal = mask_losses(mask_pred, mask_gt)
                loss_mask_focal = loss_mask_focal * 10.0

                loss = loss_l2 + loss_lpips + loss_edge + loss_mask + loss_mask_dice + loss_mask_focal

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                del sr_image, mask_pred

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_l2"] = loss_l2.detach().cpu().item()
                    logs["loss_lpips"] = loss_lpips.detach().cpu().item()
                    logs["loss_mask"] = loss_mask.detach().cpu().item()
                    logs["loss_edge"] = loss_edge.detach().cpu().item()

                    logs["loss_mask_dice"] = loss_mask_dice.detach().cpu().item()
                    logs["loss_mask_focal"] = loss_mask_focal.detach().cpu().item()

                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step != 0 and global_step % args.checkpointing_steps == 0:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    # compute validation set FID, L2, CLIP-SIM
                    if global_step != 0 and global_step % args.eval_freq == 0:
                        psnrs = []
                        ssims = []

                        for step, batch_val in enumerate(dl_val):
                            lr_image = batch_val["input"].cuda()
                            hr_image = batch_val['gt'].cuda()

                            mask_gt = batch_val["mask"].cuda()

                            B, C, H, W = lr_image.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                sr_image, mask_pred = net_pix2pix(lr_image)

                                psnr = PSNR(sr_image, hr_image)
                                ssim = SSIM(sr_image, hr_image).item()
                                psnrs.append(psnr)
                                ssims.append(ssim)

                                if step < args.num_samples_save:
                                    saver.base_url = os.path.join(args.output_dir, "results", "%05d" % global_step,
                                                                  batch_val["name"][0])
                                    saver.save_image(lr_image, "lr_image")
                                    saver.save_image(sr_image, "sr_image")
                                    saver.save_image(hr_image, "hr_image")
                                    saver.save_image(mask_gt, "mask_gt")
                                    saver.save_image(mask_pred, "mask_pred")

                                gc.collect()
                                torch.cuda.empty_cache()

                            accelerator.log(logs, step=global_step)

                        psnr = np.mean(np.array(psnrs))
                        ssim = np.mean(np.array(ssims))

                        print_in_main_process(accelerator,
                                              'Val. Epoch: {}/{}. PSNR: {:.5f}, SSIM: {:.5f}'.format(
                                                  epoch, args.num_training_epochs, psnr, ssim))


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
