import argparse
import math
import os
import random

import cv2
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils import data as data
from torchvision import transforms


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


def pad_to_max_length(image: torch.Tensor, max_length: int, fill=0.):
    from math import floor, ceil
    # Check input dims
    if image.dim() == 3:
        # (C, H, W)
        channels, height, width = image.shape
        batch = False
    elif image.dim() == 4:
        # (B, C, H, W)
        batch, channels, height, width = image.shape
    else:
        raise ValueError("Input dims must be 3 or 4.")

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

    padding = [pad_left, pad_top, pad_right, pad_bottom]

    if batch:
        padded_images = []
        for img in image:
            padded_img = F.pad(img, padding, fill=fill, padding_mode='constant')
            padded_images.append(padded_img)
        padded_image = torch.stack(padded_images)
    else:
        padded_image = F.pad(image, padding, fill=fill, padding_mode='constant')

    return padded_image


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
