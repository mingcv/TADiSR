import argparse
import json
import os
import random
from glob import glob
from typing import Tuple

import cv2
import math
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils import data as data
from torchvision import transforms
from PIL import Image, ImageFilter
from utils.degradations import circular_lowpass_kernel, random_mixed_kernels


def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_bce", default=1.0, type=float)
    parser.add_argument("--lambda_clipsim", default=5.0, type=float)

    # dataset options
    parser.add_argument("--train_folders", type=str)
    parser.add_argument("--test_folder", type=str)

    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=20, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")
    parser.add_argument("--num_samples_save", type=int, default=20, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo",
                        help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--prompt_embeddings", type=str, default=None,
                        help="Fixed CogView4 quality-prompt embedding file.")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Optional TADiSR adapter checkpoint to resume from.")
    parser.add_argument("--realce_train_list", type=str, default=None)
    parser.add_argument("--realce_eval_list", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None, )
    parser.add_argument("--variant", type=str, default=None, )
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None, )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=10_000, )
    parser.add_argument("--checkpointing_steps", type=int, default=500, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--gradient_checkpointing", action="store_true", )
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
                        )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0, )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true", )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.train_folders = args.train_folders.split(',')
    return args


def parse_args_testing(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # dataset options
    parser.add_argument("--test_folder", type=str)

    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=20, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo",
                        help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None, )
    parser.add_argument("--variant", type=str, default=None, )
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument('--selected', type=str, nargs='+')

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None, )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=10_000, )
    parser.add_argument("--checkpointing_steps", type=int, default=500, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--gradient_checkpointing", action="store_true", )
    parser.add_argument("--dataloader_num_workers", type=int, default=0, )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def parse_args_unpaired_training():
    """
    Parses command-line arguments used for configuring an unpaired session (CycleGAN-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """

    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")

    # fixed random seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_idt", default=1, type=float)
    parser.add_argument("--lambda_cycle", default=1, type=float)
    parser.add_argument("--lambda_cycle_lpips", default=10.0, type=float)
    parser.add_argument("--lambda_idt_lpips", default=1.0, type=float)

    # args for dataset and dataloader options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_img_prep", required=True)
    parser.add_argument("--val_img_prep", required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)

    # args for the model
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # args for validation and logging
    parser.add_argument("--viz_freq", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, required=True)
    parser.add_argument("--validation_steps", type=int, default=500, )
    parser.add_argument("--validation_num_images", type=int, default=-1,
                        help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    # args for the optimization options
    parser.add_argument("--learning_rate", type=float, default=5e-6, )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
    ),
                        )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.", )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # memory saving options
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Whether or not to use xformers.")

    args = parser.parse_args()
    return args


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        Itialize the paired dataset object for loading and transforming paired data samples
        from specified dataset folders.

        This constructor sets up the paths to input and output folders based on the specified 'split',
        loads the captions (or prompts) for the input images, and prepares the transformations and
        tokenizer to be applied on the data.

        Parameters:
        - dataset_folder (str): The root folder containing the dataset, expected to include
                                sub-folders for different splits (e.g., 'train_A', 'train_B').
        - split (str): The dataset split to use ('train' or 'test'), used to select the appropriate
                       sub-folders and caption files within the dataset folder.
        - image_prep (str): The image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
            captions = os.path.join(dataset_folder, "train_prompts.json")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            captions = os.path.join(dataset_folder, "test_prompts.json")
        with open(captions, "r") as f:
            self.captions = json.load(f)
        self.img_names = list(self.captions.keys())
        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.captions)

    def __getitem__(self, idx):
        """
        Retrieves a dataset item given its index. Each item consists of an input image, 
        its corresponding output image, the captions associated with the input image, 
        and the tokenized form of this caption.

        This method performs the necessary preprocessing on both the input and output images, 
        including scaling and normalization, as well as tokenizing the caption using a provided tokenizer.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        dict: A dictionary containing the following key-value pairs:
            - "output_pixel_values": a tensor of the preprocessed output image with pixel values 
            scaled to [-1, 1].
            - "conditioning_pixel_values": a tensor of the preprocessed input image with pixel values 
            scaled to [0, 1].
            - "caption": the text caption.
            - "input_ids": a tensor of the tokenized caption.

        Note:
        The actual preprocessing steps (scaling and normalization) for images are defined externally 
        and passed to this class through the `image_prep` parameter during initialization. The 
        tokenization process relies on the `tokenizer` also provided at initialization, which 
        should be compatible with the models intended to be used with this dataset.
        """
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_folder, img_name))
        output_img = Image.open(os.path.join(self.output_folder, img_name))
        caption = self.captions[img_name]

        # input images scaled to 0,1
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)
        # output images scaled to -1,1
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
        }


class PairedSRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder: str, hr_dir: str, lr_dir: str, split, image_prep):
        super().__init__()
        if split == "train":
            self.h_folder = os.path.join(dataset_folder, lr_dir)
            self.output_folder = os.path.join(dataset_folder, hr_dir)
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, lr_dir)
            self.output_folder = os.path.join(dataset_folder, hr_dir)

        self.img_names = list(os.listdir(self.output_folder))
        self.T = build_transform(image_prep)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_folder, img_name))
        output_img = Image.open(os.path.join(self.output_folder, img_name))

        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)

        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t
        }


class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        A dataset class for loading unpaired data samples from two distinct domains (source and target),
        typically used in unsupervised learning tasks like image-to-image translation.

        The class supports loading images from specified dataset folders, applying predefined image
        preprocessing transformations, and utilizing fixed textual prompts (captions) for each domain,
        tokenized using a provided tokenizer.

        Parameters:
        - dataset_folder (str): Base directory of the dataset containing subdirectories (train_A, train_B, test_A, test_B)
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): he image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")
        self.tokenizer = tokenizer
        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), "r") as f:
            self.fixed_caption_src = f.read().strip()
            self.input_ids_src = self.tokenizer(
                self.fixed_caption_src, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()
            self.input_ids_tgt = self.tokenizer(
                self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
        # find all images in the source and target folders with all IMG extensions
        self.l_imgs_src = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_src.extend(glob(os.path.join(self.source_folder, ext)))
        self.l_imgs_tgt = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_tgt.extend(glob(os.path.join(self.target_folder, ext)))
        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from the source and target domains along with their 
        corresponding tokenized captions.

        For the source domain, if the requested index is within the range of available images,
        the specific image at that index is chosen. If the index exceeds the number of source
        images, a random source image is selected. For the target domain,
        an image is always randomly selected, irrespective of the index, to maintain the 
        unpaired nature of the dataset.

        Both images are preprocessed according to the specified image transformation `T`, and normalized.
        The fixed captions for both domains
        are included along with their tokenized forms.

        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source image
            - "pixel_values_tgt": The processed target image
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
        """
        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
            img_path_src = random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])
        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }


class RealESRGANDataset(data.Dataset):
    def __init__(self, gt_folders):
        super(RealESRGANDataset, self).__init__()
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        self.paths = []

        for folder in gt_folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if os.path.splitext(file)[1].lower() in image_extensions:
                        self.paths.append(os.path.join(root, file))

        self.resize_prob = [0.2, 0.7, 0.1]
        self.resize_range = [0.15, 1.5]
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.gray_noise_prob = 0.4
        self.jpeg_range = [30, 95]

        self.second_blur_prob = 0.8
        self.resize_prob2 = [0.2, 0.7, 0.1]
        self.resize_range2 = [0.15, 1.5]
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 30]
        self.poisson_scale_range2 = [0.05, 3]
        self.gray_noise_prob2 = 0.4
        self.jpeg_range2 = [30, 95]

        self.scale = 4
        self.gt_size = 512

        # blur settings for the first degradation
        self.blur_kernel_size = 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma = [0.2, 3]
        self.betag_range = [0.5, 4]
        self.betap_range = [1, 2]
        self.sinc_prob = 0.1

        # blur settings for the second degradation
        self.blur_kernel_size2 = 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        self.sinc_prob2 = 0.1

        # a final sinc filter
        self.final_sinc_prob = 0.8

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        img_gt = cv2.imread(gt_path).astype(np.float32) / 255

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, hflip=True, rotation=True)

        # crop or pad to 400
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.gt_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
        return return_d

    def __len__(self):
        return len(self.paths)


class LSDIRDataset(data.Dataset):
    def __init__(self, dataset_file_path: str, crop_size: int = 512):
        super().__init__()
        with open(dataset_file_path, 'r') as f:
            self.img_paths = [fp.strip() for fp in f.readlines()]
        self.T = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0),
                                         interpolation=transforms.InterpolationMode.LANCZOS)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        input_img = Image.open(img_path).convert("RGB")

        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)

        fno = os.path.splitext(os.path.basename(img_path))[0]
        return {
            "fname": fno,
            "input": img_t
        }


class SRTestDataset(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder

        self.img_names = list(os.listdir(self.input_folder))
        self.T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_folder, img_name)).convert("RGB")

        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)

        return {
            "fname": os.path.splitext(img_name)[0],
            "input": img_t
        }


class CTRSynDataset(data.Dataset):
    def __init__(self, dataset_folder: str, max_no=100000):
        super().__init__()
        self.input_folder = dataset_folder
        self.max_no = max_no

    def __len__(self):
        return self.max_no

    def __getitem__(self, idx):
        name = '{}'.format(str(idx + 1).zfill(7))
        lr_path = os.path.join(self.input_folder, "sr_bicubic", name + ".png")
        gt_path = os.path.join(self.input_folder, "gt", name + ".png")
        anno_path = os.path.join(self.input_folder, "anno", name + ".pkl")

        input_img = Image.open(lr_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        input_img = F.to_tensor(input_img)
        gt_img = F.to_tensor(gt_img)
        anno = torch.load(anno_path)

        return {
            "name": name,
            "input": input_img,
            "gt": gt_img,
            "anno": anno
        }


class FTSRSynDataset(data.Dataset):
    def __init__(self, dataset_folder: str, min_no=0, max_no=500):
        super().__init__()
        self.input_folder = dataset_folder
        self.min_no = min_no
        self.max_no = max_no

    def __len__(self):
        return self.max_no - self.min_no

    def __getitem__(self, idx):
        name = '{}'.format(str(idx + self.min_no + 1).zfill(7))
        lr_path = os.path.join(self.input_folder, "sr_bicubic", name + ".png")
        gt_path = os.path.join(self.input_folder, "gt", name + ".png")
        seg_path = os.path.join(self.input_folder, "seg", name + ".png")

        input_img = Image.open(lr_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        seg_img = cv2.imread(seg_path, flags=cv2.IMREAD_ANYDEPTH)
        input_img = F.to_tensor(input_img)
        gt_img = F.to_tensor(gt_img)
        seg_img = torch.from_numpy(seg_img).float().unsqueeze(0)

        return {
            "name": name,
            "input": input_img,
            "gt": gt_img,
            "seg": seg_img
        }


class FTSRCBDataset(data.Dataset):
    def __init__(self, dataset_folder: str, min_no=0, max_no=500):
        super().__init__()
        self.input_folder = dataset_folder
        self.min_no = min_no
        self.max_no = max_no
        self.default_dict = None
        self.store_flag = False

    def __len__(self):
        return self.max_no - self.min_no

    def __getitem__(self, idx):
        name = '{}'.format(str(idx + self.min_no + 1).zfill(7))
        try:
            lr_path = os.path.join(self.input_folder, "sr_bicubic", name + ".png")
            gt_path = os.path.join(self.input_folder, "gt", name + ".png")
            mask_path = os.path.join(self.input_folder, "mask", name + ".png")
            lr_mask_path = os.path.join(self.input_folder, "lr_mask", name + ".png")

            input_img = Image.open(lr_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("L")
            lr_mask_img = Image.open(lr_mask_path).convert("L")

            input_img = F.to_tensor(input_img)
            gt_img = F.to_tensor(gt_img)
            mask_img = F.to_tensor(mask_img)
            lr_mask_img = F.to_tensor(lr_mask_img)

            res = {
                "name": name,
                "input": input_img,
                "gt": gt_img,
                "mask": mask_img,
                "lr_mask": lr_mask_img
            }

            if not self.store_flag:
                self.default_dict = res
                self.store_flag = True
                # self.default_dict['name'] = "ERROR"
            return res
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            return self.default_dict


class FTSR1024Dataset(data.Dataset):
    def __init__(self, dataset_folder: str, min_no=0, max_no=500):
        super().__init__()
        self.input_folder = dataset_folder
        self.min_no = min_no
        self.max_no = max_no
        self.default_dict = None
        self.store_flag = False

    def __len__(self):
        return self.max_no - self.min_no

    def __getitem__(self, idx):
        name = '{}'.format(str(idx + self.min_no + 1).zfill(7))
        try:
            lr_path = os.path.join(self.input_folder, "sr_bicubic", name + ".png")
            gt_path = os.path.join(self.input_folder, "gt", name + ".png")
            mask_path = os.path.join(self.input_folder, "mask", name + ".png")

            input_img = Image.open(lr_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("L")

            input_img = F.to_tensor(input_img)
            gt_img = F.to_tensor(gt_img)
            mask_img = F.to_tensor(mask_img)

            res = {
                "name": name,
                "input": input_img,
                "gt": gt_img,
                "mask": mask_img,
            }

            if not self.store_flag:
                self.default_dict = res
                self.store_flag = True
                # self.default_dict['name'] = "ERROR"
            return res
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            return self.default_dict


class FTSR768Dataset(data.Dataset):
    def __init__(self, dataset_folder: str, min_no=0, max_no=500):
        super().__init__()
        self.input_folder = dataset_folder
        self.min_no = min_no
        self.max_no = max_no
        self.default_dict = None
        self.store_flag = False
        self.transform = transforms.Compose(
            [transforms.Resize(768, interpolation=transforms.InterpolationMode.LANCZOS),
             transforms.ToTensor()]
        )

        self.transform2 = transforms.Resize(768, interpolation=transforms.InterpolationMode.LANCZOS)

    def __len__(self):
        return self.max_no - self.min_no

    def __getitem__(self, idx):
        name = '{}'.format(str(idx + self.min_no + 1).zfill(7))
        lr_path = os.path.join(self.input_folder, "sr_bicubic", name + ".png")
        gt_path = os.path.join(self.input_folder, "gt", name + ".png")
        mask_path = os.path.join(self.input_folder, "mask", name + ".png")

        input_img = Image.open(lr_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        input_img = self.transform(input_img)
        gt_img = self.transform(gt_img)
        mask_img_tensor = self.transform(mask_img)
        mask_ori = self.transform2(mask_img)
        mask_ori = np.expand_dims(np.array(mask_ori), axis=-1)
        mask_ori = np.repeat(mask_ori, 3, axis=-1)

        res = {
            "name": name,
            "input": input_img,
            "gt": gt_img,
            "mask": mask_img_tensor,
            "mask_ori": mask_ori
        }
        return res

class FTSR512Dataset(data.Dataset):
    def __init__(self, dataset_folder: str, min_no=0, max_no=500):
        super().__init__()
        self.input_folder = dataset_folder
        self.min_no = min_no
        self.max_no = max_no
        self.default_dict = None
        self.store_flag = False
        self.transform = transforms.Compose(
            [transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
             transforms.ToTensor()]
        )

        self.transform2 = transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS)

    def __len__(self):
        return self.max_no - self.min_no

    def __getitem__(self, idx):
        name = '{}'.format(str(idx + self.min_no + 1).zfill(7))
        lr_path = os.path.join(self.input_folder, "sr_bicubic", name + ".png")
        gt_path = os.path.join(self.input_folder, "gt", name + ".png")
        mask_path = os.path.join(self.input_folder, "mask", name + ".png")

        input_img = Image.open(lr_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        input_img = self.transform(input_img)
        gt_img = self.transform(gt_img)
        mask_img_tensor = self.transform(mask_img)
        mask_ori = self.transform2(mask_img)
        mask_ori = np.expand_dims(np.array(mask_ori), axis=-1)
        mask_ori = np.repeat(mask_ori, 3, axis=-1)

        res = {
            "name": name,
            "input": input_img,
            "gt": gt_img,
            "mask": mask_img_tensor,
            "mask_ori": mask_ori
        }
        return res


class FTSRCBDataset2(data.Dataset):
    def __init__(self, dataset_folder: str, min_no=0, max_no=500):
        super().__init__()
        self.input_folder = dataset_folder
        self.min_no = min_no
        self.max_no = max_no
        self.default_dict = None
        self.store_flag = False

    def __len__(self):
        return self.max_no - self.min_no

    def __getitem__(self, idx):
        name = '{}'.format(str(idx + self.min_no + 1).zfill(7))
        try:
            lr_path = os.path.join(self.input_folder, "sr_bicubic", name + ".png")
            gt_path = os.path.join(self.input_folder, "gt", name + ".png")
            mask_path = os.path.join(self.input_folder, "mask", name + ".png")
            lr_mask_path = os.path.join(self.input_folder, "rendered_mask", name + ".png")

            input_img = Image.open(lr_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("L")
            lr_mask_img = Image.open(lr_mask_path).convert("L")

            input_img = F.to_tensor(input_img)
            gt_img = F.to_tensor(gt_img)
            mask_img = F.to_tensor(mask_img)
            lr_mask_img = F.to_tensor(lr_mask_img)

            res = {
                "name": name,
                "input": input_img,
                "gt": gt_img,
                "mask": mask_img,
                "lr_mask": lr_mask_img
            }

            if not self.store_flag:
                self.default_dict = res
                self.store_flag = True
                # self.default_dict['name'] = "ERROR"
            return res
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            return self.default_dict


class ResizeLongestSide_ToTensor(object):
    def __init__(self, target_length=512):
        self.target_length = target_length

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return [newh, neww]

    def __call__(self, image, value=128, grayscale=False):
        target_size = self.get_preprocess_shape(*image.shape[:-1], self.target_length)
        padding_h = max(self.target_length - target_size[0], 0)
        padding_w = max(self.target_length - target_size[1], 0)

        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        image = cv2.copyMakeBorder(image, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=(value, value, value))
        if not grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.as_tensor(image, dtype=torch.float32) / 255.
        image = image.permute(2, 0, 1)
        return image


class FTSRRealCEDataset(data.Dataset):
    def __init__(self, dataset_folder: str, img_list_file):
        super().__init__()
        self.input_folder = dataset_folder
        self.default_dict = None
        self.store_flag = False
        with open(img_list_file, "r") as f:
            img_list = f.readlines()
        self.img_list = [line.strip() for line in img_list]
        self.img_list.sort()
        self.transform = ResizeLongestSide_ToTensor()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        name = os.path.splitext(self.img_list[idx])[0]
        try:
            lr_path = os.path.join(self.input_folder, "13mm", name + ".JPG")
            gt_path = os.path.join(self.input_folder, "52mm", name + ".JPG")
            lr_mask_path = os.path.join(self.input_folder, "render-13mm", name + ".png")

            input_img = cv2.imread(lr_path)
            gt_img = cv2.imread(gt_path)
            lr_mask_img = Image.open(lr_mask_path).convert("L")

            input_img = self.transform(input_img)
            gt_img = self.transform(gt_img)
            lr_mask_img = F.to_tensor(lr_mask_img)

            res = {
                "name": name,
                "input": input_img,
                "gt": gt_img,
                "lr_mask": lr_mask_img
            }

            if not self.store_flag:
                self.default_dict = res
                self.store_flag = True
                # self.default_dict['name'] = "ERROR"
            return res
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            return self.default_dict


class FTSRRealCEDataset2(data.Dataset):
    def __init__(self, dataset_folder: str, img_list_file):
        super().__init__()
        self.input_folder = dataset_folder
        self.default_dict = None
        self.store_flag = False
        with open(img_list_file, "r") as f:
            img_list = f.readlines()
        self.img_list = [line.strip() for line in img_list]
        self.img_list.sort()
        self.transform = ResizeLongestSide_ToTensor()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        name = os.path.splitext(self.img_list[idx])[0]
        lr_path = os.path.join(self.input_folder, "13mm_aligned_1024", name + ".JPG")
        gt_path = os.path.join(self.input_folder, "52mm_aligned_1024", name + ".JPG")
        lr_mask_path = os.path.join(self.input_folder, "mask-52mm-bts-1024-v2", name + ".png")

        input_img = cv2.imread(lr_path)
        gt_img = cv2.imread(gt_path)
        lr_mask_img = cv2.imread(lr_mask_path)
        # print(lr_mask_img.shape)

        input_img = self.transform(input_img, 128, grayscale=False)
        gt_img = self.transform(gt_img, 128, grayscale=False)
        lr_mask_img = self.transform(lr_mask_img, 0, grayscale=True)[:1]

        res = {
            "name": name,
            "input": input_img,
            "gt": gt_img,
            "lr_mask": lr_mask_img
        }

        return res


class ResizeLongestSide:
    def __init__(self, target_length: int, interpolation=transforms.InterpolationMode.LANCZOS):
        self.target_length = target_length
        self.interpolation = interpolation

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f'Input should be a PIL Image. Got {type(img)}')

        width, height = img.size
        if width >= height:
            scale = self.target_length / width
        else:
            scale = self.target_length / height

        new_width = int(math.ceil(width * scale) // 32 * 32)
        new_height = int(math.ceil(height * scale) // 32 * 32)

        return F.resize(img, (new_height, new_width), interpolation=self.interpolation)

    @staticmethod
    def get_new_size(shape, target_length):
        width, height = shape
        if width >= height:
            scale = target_length / width
        else:
            scale = target_length / height

        new_width = int(math.ceil(width * scale) // 32 * 32)
        new_height = int(math.ceil(height * scale) // 32 * 32)
        return new_width, new_height

    def __repr__(self):
        return f"{self.__class__.__name__}(target_length={self.target_length}, interpolation={self.interpolation})"


class ResizeRound:
    def __init__(self, interpolation=transforms.InterpolationMode.LANCZOS):
        self.interpolation = interpolation

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f'Input should be a PIL Image. Got {type(img)}')

        width, height = img.size

        new_width = int(math.ceil(width) // 32 * 32)
        new_height = int(math.ceil(height) // 32 * 32)

        return F.resize(img, (new_height, new_width), interpolation=self.interpolation)


class FTSRRealCEDataset512(data.Dataset):
    def __init__(self, dataset_folder: str, img_list_file):
        super().__init__()
        self.input_folder = dataset_folder
        self.default_dict = None
        self.store_flag = False
        with open(img_list_file, "r") as f:
            img_list = f.readlines()
        self.img_list = [line.strip() for line in img_list]
        self.img_list.sort()

        self.transform = transforms.Compose(
            [ResizeLongestSide(512, interpolation=transforms.InterpolationMode.LANCZOS),
             transforms.ToTensor()]
        )

        self.transform2 = ResizeLongestSide(512, interpolation=transforms.InterpolationMode.LANCZOS)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        name = os.path.splitext(self.img_list[idx])[0]
        lr_path = os.path.join(self.input_folder, "13mm_aligned_1024", name + ".JPG")
        gt_path = os.path.join(self.input_folder, "52mm_aligned_1024", name + ".JPG")
        mask_path = os.path.join(self.input_folder, "mask-52mm-bts-1024-v2", name + ".png")

        input_img = Image.open(lr_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        input_img = self.transform(input_img)
        gt_img = self.transform(gt_img)
        mask_img_tensor = self.transform(mask_img)
        mask_ori = self.transform2(mask_img)
        mask_ori = np.expand_dims(np.array(mask_ori), axis=-1)
        mask_ori = np.repeat(mask_ori, 3, axis=-1)

        res = {
            "name": name,
            "input": input_img,
            "gt": gt_img,
            "mask": mask_img_tensor,
            "mask_ori": mask_ori
        }
        return res


class FTSRRealCEDatasetUA768(data.Dataset):
    def __init__(self, dataset_folder: str, img_list_file):
        super().__init__()
        self.input_folder = dataset_folder
        self.default_dict = None
        self.store_flag = False
        with open(img_list_file, "r") as f:
            img_list = f.readlines()
        self.img_list = [line.strip() for line in img_list]
        self.img_list.sort()

        self.transform = transforms.Compose(
            [ResizeLongestSide(768, interpolation=transforms.InterpolationMode.LANCZOS),
             transforms.ToTensor()]
        )

        self.transform2 = ResizeLongestSide(768, interpolation=transforms.InterpolationMode.LANCZOS)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        name = os.path.splitext(self.img_list[idx])[0]
        lr_path = os.path.join(self.input_folder, "13mm", name + ".JPG")
        gt_path = os.path.join(self.input_folder, "52mm", name + ".JPG")

        input_img = Image.open(lr_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        input_img = self.transform(input_img)
        gt_img = self.transform(gt_img)

        res = {
            "name": name,
            "input": input_img,
            "gt": gt_img,
        }
        return res


class FTSRRealCEDatasetScaled(data.Dataset):
    def __init__(self, dataset_folder: str, img_list_file):
        super().__init__()
        self.input_folder = dataset_folder
        self.default_dict = None
        self.store_flag = False
        with open(img_list_file, "r") as f:
            img_list = f.readlines()
        self.img_list = [line.strip() for line in img_list]
        self.img_list.sort()

        self.interpolation = transforms.InterpolationMode.LANCZOS
        self.transform = transforms.ToTensor()
        self.transform2 = ResizeLongestSide(496, interpolation=self.interpolation)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        name = os.path.splitext(self.img_list[idx])[0]
        lr_path = os.path.join(self.input_folder, "13mm_aligned_1024", name + ".JPG")
        gt_path = os.path.join(self.input_folder, "52mm_aligned_1024", name + ".JPG")
        mask_path = os.path.join(self.input_folder, "mask-52mm-bts-1024-v2", name + ".png")

        input_img = Image.open(lr_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        orig_width, orig_height = ResizeLongestSide.get_new_size((input_img.width, input_img.height), 496)

        scale = 0
        new_size = None
        # 是否上采样
        if random.random() < 0.7:
            scale = random.uniform(2, 6)
            new_size = (int(orig_height * scale), int(orig_width * scale))
            input_img = F.resize(input_img, new_size, interpolation=self.interpolation)
            gt_img = F.resize(gt_img, new_size, interpolation=self.interpolation)
            mask_img = F.resize(mask_img, new_size, interpolation=self.interpolation)
        else:
            input_img = self.transform2(input_img)
            gt_img = self.transform2(gt_img)
            mask_img = self.transform2(mask_img)
        # 裁剪 patch
        x_max = input_img.width - orig_width
        y_max = input_img.height - orig_height

        resized_size = (input_img.width, input_img.height)
        if x_max > 0 and y_max > 0:
            x = random.randint(0, x_max)
            y = random.randint(0, y_max)
            input_img = input_img.crop((x, y, x + orig_width, y + orig_height))
            gt_img = gt_img.crop((x, y, x + orig_width, y + orig_height))
            mask_img = mask_img.crop((x, y, x + orig_width, y + orig_height))

        # 转 tensor
        input_img = self.transform(input_img)
        gt_img = self.transform(gt_img)
        mask_img_tensor = self.transform(mask_img)
        # print((orig_width, orig_height), scale, new_size, resized_size, (x_max, y_max), input_img.shape)

        mask_ori = self.transform2(mask_img)
        mask_ori = np.expand_dims(np.array(mask_ori), axis=-1)
        mask_ori = np.repeat(mask_ori, 3, axis=-1)

        return {
            "name": name,
            "input": input_img,
            "gt": gt_img,
            "mask": mask_img_tensor,
            "mask_ori": mask_ori
        }


class FTSRRealTestDataset(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.default_dict = None
        self.store_flag = False
        self.img_list = os.listdir(dataset_folder)
        self.img_list.sort()

        self.transform = transforms.Compose(
            [ResizeRound(interpolation=transforms.InterpolationMode.LANCZOS),
             transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        fname = self.img_list[idx]
        fno = os.path.splitext(fname)[0]
        lr_path = os.path.join(self.input_folder, fname)
        input_img = Image.open(lr_path).convert("RGB")
        input_img = self.transform(input_img)
        res = {
            "name": fno,
            "input": input_img
        }
        return res


class FTSRRealTestDataset768(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.default_dict = None
        self.store_flag = False
        self.img_list = os.listdir(dataset_folder)
        self.img_list.sort()

        self.transform = transforms.Compose(
            [ResizeLongestSide(768, interpolation=transforms.InterpolationMode.LANCZOS),
             transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        fname = self.img_list[idx]
        fno = os.path.splitext(fname)[0]
        lr_path = os.path.join(self.input_folder, fname)
        input_img = Image.open(lr_path).convert("RGB")
        input_img = self.transform(input_img)
        res = {
            "name": fno,
            "input": input_img
        }
        return res


class FTSRRealTestDataset(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.default_dict = None
        self.store_flag = False
        self.img_list = os.listdir(dataset_folder)
        self.img_list.sort()

        self.transform = transforms.Compose(
            [ResizeRound(interpolation=transforms.InterpolationMode.LANCZOS),
             transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        fname = self.img_list[idx]
        fno = os.path.splitext(fname)[0]
        lr_path = os.path.join(self.input_folder, fname)
        input_img = Image.open(lr_path).convert("RGB")
        input_img = self.transform(input_img)
        res = {
            "name": fno,
            "input": input_img
        }
        return res


class FTSRRealTestDataset512(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.default_dict = None
        self.store_flag = False
        self.img_list = os.listdir(dataset_folder)
        self.img_list.sort()

        self.transform = transforms.Compose(
            [ResizeLongestSide(512, interpolation=transforms.InterpolationMode.LANCZOS),
             transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        fname = self.img_list[idx]
        fno = os.path.splitext(fname)[0]
        lr_path = os.path.join(self.input_folder, fname)
        input_img = Image.open(lr_path).convert("RGB")
        input_img = self.transform(input_img)
        res = {
            "name": fno,
            "input": input_img
        }
        return res


class FTSRRealDataset(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.img_names = os.listdir(os.path.join(self.input_folder, "lr_resize_deg_pil2"))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        lr_path = os.path.join(self.input_folder, "lr_resize_deg_pil2", name)
        hr_path = os.path.join(self.input_folder, "lr", name)
        seg_path = os.path.join(self.input_folder, "seg", name.split('.')[0] + ".png")

        input_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        seg_img = cv2.imread(seg_path, flags=cv2.IMREAD_ANYDEPTH)
        input_img = F.to_tensor(input_img)
        hr_img = F.to_tensor(hr_img)
        seg_img = torch.from_numpy(seg_img).float().unsqueeze(0)

        input_img = F.resize(input_img, size=[512, 512], antialias=True)
        hr_img = F.resize(hr_img, size=[512, 512], antialias=True)
        seg_img = F.resize(seg_img, size=[512, 512], antialias=True)
        return {
            "name": name,
            "input": input_img,
            "hr": hr_img,
            "seg": seg_img
        }


class FTSRRealDataset2(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.img_names = os.listdir(os.path.join(self.input_folder, "sr_bicubic"))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        lr_path = os.path.join(self.input_folder, "sr_bicubic", name)
        hr_path = os.path.join(self.input_folder, "sr_bicubic", name)
        mask_path = os.path.join(self.input_folder, "mask", name.split('.')[0] + ".png")

        input_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        input_img = F.to_tensor(input_img)
        hr_img = F.to_tensor(hr_img)
        mask_img = F.to_tensor(mask_img)

        input_img = F.resize(input_img, size=[512, 512])
        hr_img = F.resize(hr_img, size=[512, 512])
        mask_img = F.resize(mask_img, size=[512, 512])
        return {
            "name": name,
            "input": input_img,
            "gt": hr_img,
            "mask": mask_img
        }


class FTSRRealDataset4(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.img_names = os.listdir(os.path.join(self.input_folder, "sr_bicubic"))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        lr_path = os.path.join(self.input_folder, "sr_bicubic", name)
        hr_path = os.path.join(self.input_folder, "gt", name)
        mask_path = os.path.join(self.input_folder, "mask", name.split('.')[0] + ".png")
        lr_mask_path = os.path.join(self.input_folder, "lr_mask", name.split('.')[0] + ".png")

        input_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        lr_mask_img = Image.open(lr_mask_path).convert("L")

        input_img = F.to_tensor(input_img)
        hr_img = F.to_tensor(hr_img)
        mask_img = F.to_tensor(mask_img)
        lr_mask_img = F.to_tensor(lr_mask_img)

        input_img = F.resize(input_img, size=[512, 512])
        hr_img = F.resize(hr_img, size=[512, 512])
        mask_img = F.resize(mask_img, size=[512, 512])
        lr_mask_img = F.resize(lr_mask_img, size=[512, 512])
        return {
            "name": name,
            "input": input_img,
            "gt": hr_img,
            "mask": mask_img,
            "lr_mask": lr_mask_img
        }


class FTSRRealDataset5(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.img_names = os.listdir(os.path.join(self.input_folder, "sr_bicubic"))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        lr_path = os.path.join(self.input_folder, "sr_bicubic", name)
        hr_path = os.path.join(self.input_folder, "gt", name)
        lr_mask_path = os.path.join(self.input_folder, "lr_mask", name.split('.')[0] + ".png")

        input_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        lr_mask_img = Image.open(lr_mask_path).convert("L")

        input_img = F.to_tensor(input_img)
        hr_img = F.to_tensor(hr_img)
        lr_mask_img = F.to_tensor(lr_mask_img)

        input_img = F.resize(input_img, size=[512, 512])
        hr_img = F.resize(hr_img, size=[512, 512])
        lr_mask_img = F.resize(lr_mask_img, size=[512, 512])
        return {
            "name": name,
            "gt": hr_img,
            "input": input_img,
            "lr_mask": lr_mask_img
        }


class FTSRRealDataset6(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.img_names = os.listdir(os.path.join(self.input_folder, "sr_bicubic"))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        lr_path = os.path.join(self.input_folder, "sr_bicubic", name)
        hr_path = os.path.join(self.input_folder, "gt", name)
        lr_mask_path = os.path.join(self.input_folder, "rendered_mask", name.split('.')[0] + ".png")

        input_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        lr_mask_img = Image.open(lr_mask_path).convert("L")

        input_img = F.to_tensor(input_img)
        hr_img = F.to_tensor(hr_img)
        lr_mask_img = F.to_tensor(lr_mask_img)

        input_img = F.resize(input_img, size=[512, 512])
        hr_img = F.resize(hr_img, size=[512, 512])
        lr_mask_img = F.resize(lr_mask_img, size=[512, 512])
        return {
            "name": name,
            "gt": hr_img,
            "input": input_img,
            "lr_mask": lr_mask_img
        }


class FTSRRealDatasetTest(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.img_names = os.listdir(self.input_folder)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        lr_path = os.path.join(self.input_folder, name)
        input_img = Image.open(lr_path).convert("RGB")
        input_img = F.to_tensor(input_img)
        resized_image = resize_aspect_ratio(input_img, 512, interpolation=F.InterpolationMode.BILINEAR)
        padded_image = pad_to_max_length(resized_image, 512, fill=0.)
        return {
            "name": name,
            "input": padded_image
        }


def resize_aspect_ratio(image: torch.Tensor, max_length: int,
                        interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR) -> torch.Tensor:
    """
    将输入的图像张量按长边符合给定长度并保持宽高比进行缩放。

    Args:
        image (torch.Tensor): 输入图像张量，形状为 (C, H, W) 或 (B, C, H, W)。
        max_length (int): 长边的目标长度。
        interpolation (InterpolationMode): 插值方式，默认双线性插值。

    Returns:
        torch.Tensor: 缩放后的图像张量，保持原始宽高比。
    """
    # 检查输入维度
    if image.dim() == 3:
        # (C, H, W)
        channels, height, width = image.shape
    elif image.dim() == 4:
        # (B, C, H, W)
        batch, channels, height, width = image.shape
    else:
        raise ValueError("输入的图像张量维度必须是 3 或 4。")

    # 确定当前长边和短边
    if height >= width:
        new_height = max_length
        new_width = int(round((max_length / height) * width))
    else:
        new_width = max_length
        new_height = int(round((max_length / width) * height))

    # 使用F.resize进行缩放
    resized_image = F.resize(image, size=[new_height, new_width], interpolation=interpolation)

    return resized_image


def pad_to_max_length(image: torch.Tensor, max_length: int, fill=0.):
    from math import floor, ceil
    # 检查输入维度
    if image.dim() == 3:
        # (C, H, W)
        channels, height, width = image.shape
        batch = False
    elif image.dim() == 4:
        # (B, C, H, W)
        batch, channels, height, width = image.shape
    else:
        raise ValueError("输入的图像张量维度必须是 3 或 4。")

    # 计算需要的填充量
    if height < max_length:
        pad_vert = max_length - height
        pad_top = floor(pad_vert / 2)
        pad_bottom = ceil(pad_vert / 2)
    else:
        pad_top = pad_bottom = 0

    if width < max_length:
        pad_horiz = max_length - width
        pad_left = floor(pad_horiz / 2)
        pad_right = ceil(pad_horiz / 2)
    else:
        pad_left = pad_right = 0

    padding = [pad_left, pad_top, pad_right, pad_bottom]  # 顺序为 (左, 上, 右, 下)

    if batch:
        # 对于批量图像，逐个应用填充
        padded_images = []
        for img in image:
            padded_img = F.pad(img, padding, fill=fill, padding_mode='constant')
            padded_images.append(padded_img)
        padded_image = torch.stack(padded_images)
    else:
        # 单张图像
        padded_image = F.pad(image, padding, fill=fill, padding_mode='constant')

    return padded_image


def get_bounding_box(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    根据掩码计算外接的边界框。

    Args:
        mask (torch.Tensor): 二值掩码张量，形状为 (H, W)。

    Returns:
        Tuple[int, int, int, int]: 边界框坐标 (left, top, right, bottom)。
    """
    assert mask.dim() == 2, "掩码必须是二维张量 (H, W)。"
    # 找到非零点的坐标
    non_zero = mask.nonzero(as_tuple=False)
    if non_zero.numel() == 0:
        # 如果掩码全为零，返回整个图像
        return 0, 0, mask.shape[1], mask.shape[0]
    y_min = torch.min(non_zero[:, 0]).item()
    y_max = torch.max(non_zero[:, 0]).item()
    x_min = torch.min(non_zero[:, 1]).item()
    x_max = torch.max(non_zero[:, 1]).item()
    return x_min, y_min, x_max, y_max


def crop_image_and_mask(image: torch.Tensor, mask: torch.Tensor, bbox: Tuple[int, int, int, int]) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    根据边界框裁剪图像和掩码。

    Args:
        image (torch.Tensor): 输入图像张量，形状为 (C, H, W)。
        mask (torch.Tensor): 输入掩码张量，形状为 (H, W)。
        bbox (Tuple[int, int, int, int]): 边界框坐标 (left, top, right, bottom)。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 裁剪后的图像和掩码。
    """
    left, top, right, bottom = bbox
    cropped_image = image[:, top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]
    return cropped_image, cropped_mask


def apply_gaussian_blur(image: torch.Tensor, kernel_size=5, sigma: float = 1.0) -> torch.Tensor:
    """
    对图像应用高斯模糊。

    Args:
        image (torch.Tensor): 输入图像张量，形状为 (C, H, W) 或 (B, C, H, W)。
        kernel_size (int): 高斯核大小。必须是正奇数。
        sigma (float): 高斯核的标准差。

    Returns:
        torch.Tensor: 应用高斯模糊后的图像张量。
    """
    # 确保kernel_size是奇数
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size 必须是正奇数。")

    # 应用高斯模糊
    blurred_image = F.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
    return blurred_image


def process_image_and_mask(
        image: torch.Tensor,
        mask: torch.Tensor,
        target_length: int,
        fill_image=0,
        fill_mask=0,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据掩码计算边界框，裁剪图像和掩码，仅保留文本区域，应用高斯模糊，然后进行缩放和填充。

    Args:
        image (torch.Tensor): 输入图像张量，形状为 (C, H, W)。
        mask (torch.Tensor): 输入掩码张量，形状为 (H, W)。
        target_length (int): 规范化的目标长度（长边和填充后的边长）。
        fill_image (int, tuple): 图像填充的像素值。默认0。
        fill_mask (int, tuple): 掩码填充的像素值。默认0。
        blur_kernel_size (int): 高斯模糊核大小。默认5。
        blur_sigma (float): 高斯模糊标准差。默认1.0。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 规范化后的图像和掩码，形状为 (C, target_length, target_length) 和 (1, target_length, target_length)。
    """
    # 1. 计算边界框
    bbox = get_bounding_box(mask)
    print(f"Bounding Box: {bbox}")

    # 2. 裁剪图像和掩码
    cropped_image, cropped_mask = crop_image_and_mask(image, mask, bbox)

    # 3. 应用高斯模糊到裁剪后的图像
    blurred_image = apply_gaussian_blur(cropped_image, kernel_size=blur_kernel_size, sigma=blur_sigma)

    # 4. 扩展掩码维度以适应 F.resize
    cropped_mask = cropped_mask.unsqueeze(0)  # 形状: (1, H, W)

    # 5. 缩放图像和掩码
    resized_image = resize_aspect_ratio(blurred_image, target_length, interpolation=F.InterpolationMode.BILINEAR)
    resized_mask = resize_aspect_ratio(cropped_mask, target_length, interpolation=F.InterpolationMode.NEAREST)

    # 6. 填充图像和掩码
    padded_image = pad_to_max_length(resized_image, target_length, fill=fill_image)
    padded_mask = pad_to_max_length(resized_mask, target_length, fill=fill_mask)

    return padded_image, padded_mask


class BTSMaskDataset(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.img_names = os.listdir(os.path.join(self.input_folder, "image"))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        base_name = os.path.splitext(name)[0]
        lr_path = os.path.join(self.input_folder, "image", base_name + ".jpg")
        lr_mask_path = os.path.join(self.input_folder, "semantic_label", base_name + ".png")

        input_img = Image.open(lr_path).convert("RGB")
        lr_mask_img = Image.open(lr_mask_path).convert("L")

        input_img = F.to_tensor(input_img)
        lr_mask_img = F.to_tensor(lr_mask_img)
        lr_mask_img = (lr_mask_img[0] > 0).float()
        processed_image, processed_mask = process_image_and_mask(
            image=input_img,
            mask=lr_mask_img,
            target_length=512,
            fill_image=0,
            fill_mask=0,
            blur_kernel_size=11,
            blur_sigma=4
        )

        return {
            "name": name,
            "input": processed_image,
            "lr_mask": processed_mask
        }


class FTSRRealDatasetDilation(data.Dataset):
    def __init__(self, dataset_folder: str, dilation_radius: int = 3):
        super().__init__()
        self.input_folder = dataset_folder
        self.img_names = os.listdir(os.path.join(self.input_folder, "sr_bicubic"))
        self.dilation_radius = dilation_radius

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.img_names)

    def __getitem__(self, idx):
        """Fetches the input image, ground truth, and dilated mask."""
        name = self.img_names[idx]
        lr_path = os.path.join(self.input_folder, "sr_bicubic", name)
        hr_path = os.path.join(self.input_folder, "sr_bicubic", name)
        mask_path = os.path.join(self.input_folder, "mask", name.split('.')[0] + ".png")

        # Load the images
        input_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        # Apply dilation to the mask
        mask_img = mask_img.filter(ImageFilter.MaxFilter(self.dilation_radius))

        # Convert images to tensors
        input_img = transforms.ToTensor()(input_img)
        hr_img = transforms.ToTensor()(hr_img)
        mask_img = transforms.ToTensor()(mask_img)

        # Resize all images to 512x512
        input_img = transforms.Resize((512, 512))(input_img)
        hr_img = transforms.Resize((512, 512))(hr_img)
        mask_img = transforms.Resize((512, 512))(mask_img)

        return {
            "name": name,
            "input": input_img,
            "gt": hr_img,
            "mask": mask_img
        }


class FTSRRealInverseDataset(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.img_names = sorted(os.listdir(os.path.join(self.input_folder, "sr_bicubic")))
        self.mask_names = sorted(os.listdir(os.path.join(self.input_folder, "mask")), reverse=True)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        name_mask = self.mask_names[idx]
        lr_path = os.path.join(self.input_folder, "sr_bicubic", name)
        hr_path = os.path.join(self.input_folder, "sr_bicubic", name)
        mask_path = os.path.join(self.input_folder, "mask", name_mask.split('.')[0] + ".png")

        input_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        input_img = F.to_tensor(input_img)
        hr_img = F.to_tensor(hr_img)
        mask_img = F.to_tensor(mask_img)

        input_img = F.resize(input_img, size=[512, 512])
        hr_img = F.resize(hr_img, size=[512, 512])
        mask_img = F.resize(mask_img, size=[512, 512])
        return {
            "name": name,
            "input": input_img,
            "gt": hr_img,
            "mask": mask_img
        }


class FTSRMaskDataset(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.img_names = os.listdir(os.path.join(self.input_folder, "lr_mask"))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        lr_path = os.path.join(self.input_folder, "lr_mask", name)
        hr_path = os.path.join(self.input_folder, "mask", name)

        input_img = Image.open(lr_path).convert("L")
        hr_img = Image.open(hr_path).convert("L")

        input_img = F.to_tensor(input_img)
        hr_img = F.to_tensor(hr_img)

        input_img = F.resize(input_img, size=[512, 512])
        hr_img = F.resize(hr_img, size=[512, 512])
        return {
            "name": name,
            "input": input_img,
            "gt": hr_img,
        }
