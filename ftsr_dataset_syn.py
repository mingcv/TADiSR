import glob
import json
import os
import random

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils.img_process_util import filter2D
from basicsr.utils import DiffJPEG
from torch.utils.data import Dataset
from tqdm import tqdm
import saver

jpeger = DiffJPEG(differentiable=False).cuda()


class RealESRGANDataset(data.Dataset):
    def __init__(self, opt):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 1024

        # support multiple type of data: file path and meta data, remove support of lmdb
        self.gt_data = SyntheticDataset(
            large_image_path=opt['large_image_path'],
            ctr_dataset_path=opt['ctr_dataset_path'],
            ctr_json_file=opt['ctr_json_file'],
            bts_dataset_path=opt['bts_dataset_path'],
            textseg_dataset_path=opt['textseg_dataset_path'],
            output_size=(self.crop_size, self.crop_size),
            get_pil_image=False
        )
        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21

        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
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
        sinc_kernel = self.pulse_tensor

        data_item = self.gt_data[index]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': data_item['image'],
                    'mask': data_item['mask'],
                    'kernel1': kernel,
                    'kernel2': kernel2,
                    'sinc_kernel': sinc_kernel}
        return return_d

    def __len__(self):
        return len(self.gt_data)


def realesrgan_degradation(batch, args_degradation, sf=4):
    im_gt = batch['gt'].cuda()
    mask = batch['mask'].cuda()
    im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
    kernel1 = batch['kernel1'].cuda()

    ori_h, ori_w = im_gt.size()[2:4]

    updown_type = random.choices(['up', 'down', 'keep'], args_degradation['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, args_degradation['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(args_degradation['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(im_gt, scale_factor=scale, mode=mode)
    # ----------------------- The first degradation process ----------------------- #
    if random.random() > 0.5:
        # blur
        out = filter2D(out, kernel1)
        # add noise
        gray_noise_prob = args_degradation['gray_noise_prob']
        if random.random() < args_degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=args_degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=args_degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
    else:
        gray_noise_prob = args_degradation['gray_noise_prob']
        if random.random() < args_degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                im_gt,
                sigma_range=args_degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                im_gt,
                scale_range=args_degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        out = filter2D(out, kernel1)

    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], args_degradation['resize_prob2'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, args_degradation['resize_range2'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(args_degradation['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['bilinear', 'bicubic'])
    out = F.interpolate(
        out, size=(int(ori_h / args_degradation['scale'] * scale),
                   int(ori_w / args_degradation['scale'] * scale)),
        mode=mode)
    # add noise
    gray_noise_prob = args_degradation['gray_noise_prob2']
    if np.random.uniform() < args_degradation['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=args_degradation['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=args_degradation['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)

    if np.random.uniform() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // args_degradation['scale'], ori_w // args_degradation['scale']),
                            mode=mode)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // args_degradation['scale'], ori_w // args_degradation['scale']),
                            mode=mode)

    # clamp and round
    im_lq = torch.clamp(out, 0, 1.0)

    # random crop
    gt_size = args_degradation['gt_size']
    im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, sf)
    lq, gt = im_lq, im_gt

    gt = torch.clamp(gt, 0, 1)
    lq = torch.clamp(lq, 0, 1)

    return lq, gt, mask


def get_dataset_settings():
    args_training_dataset = {}
    args_training_dataset['large_image_path'] = "[large_image_path]"
    args_training_dataset['ctr_dataset_path'] = "[ctr_dataset_path]"
    args_training_dataset['ctr_json_file'] = "[ctr_json_file]"
    args_training_dataset['bts_dataset_path'] = "[bts_dataset_path]"
    args_training_dataset['textseg_dataset_path'] = "[textseg_dataset_path]"
    args_training_dataset['min_scale'] = 0.1

    args_training_dataset['queue_size'] = 160
    args_training_dataset['crop_size'] = 1024

    args_training_dataset['blur_kernel_size'] = 7
    args_training_dataset['kernel_list'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso',
                                            'plateau_aniso']
    args_training_dataset['kernel_prob'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    args_training_dataset['sinc_prob'] = 0.1
    args_training_dataset['blur_sigma'] = [0.1, 0.5]
    args_training_dataset['betag_range'] = [0.5, 1.1]
    args_training_dataset['betap_range'] = [1, 1.1]

    args_training_dataset['blur_kernel_size2'] = 7
    args_training_dataset['kernel_list2'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso',
                                             'plateau_aniso']
    args_training_dataset['kernel_prob2'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    args_training_dataset['blur_sigma2'] = [0.1, 0.5]
    args_training_dataset['betag_range2'] = [0.5, 1.1]
    args_training_dataset['betap_range2'] = [1, 1.1]
    return args_training_dataset


def get_degradation_settings():
    args_degradation = {}
    # the first degradation process
    args_degradation['scale'] = 4
    args_degradation['resize_prob'] = [0.2, 0.7, 0.1]
    args_degradation['resize_range'] = [0.2, 1.5]
    args_degradation['gaussian_noise_prob'] = 0.5
    args_degradation['noise_range'] = [15, 50]
    args_degradation['poisson_scale_range'] = [0.25, 1.25]
    args_degradation['gray_noise_prob'] = 0.2

    args_degradation['resize_prob2'] = [0.3, 0.4, 0.3]
    args_degradation['resize_range2'] = [0.8, 1.2]
    args_degradation['gaussian_noise_prob2'] = 0.5
    args_degradation['noise_range2'] = [1, 25]
    args_degradation['poisson_scale_range2'] = [0.25, 1.25]
    args_degradation['gray_noise_prob2'] = 0.4
    args_degradation['jpeg_range2'] = [60, 100]

    args_degradation['gt_size'] = 1024
    args_degradation['no_degradation_prob'] = 0.05
    return args_degradation


def load_image_paths(dataset_path):
    if isinstance(dataset_path, str):
        if os.path.isdir(dataset_path):
            return glob.glob(os.path.join(dataset_path, '*'))
        elif os.path.isfile(dataset_path):
            with open(dataset_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            raise ValueError(f"Invalid path: {dataset_path}")
    else:
        return dataset_path


def load_lsdir_paths(dataset_path):
    image_paths = []
    for subdir in tqdm(os.listdir(dataset_path), desc="loading lsdir data..."):
        subdir_path = os.path.join(dataset_path, subdir)
        for fn in os.listdir(subdir_path):
            fp = os.path.join(subdir_path, fn)
            image_paths.append(fp)
        if len(image_paths) > 1000:
            break

    return image_paths


def load_ctr_mask_paths(dataset_path, json_file):
    with open(json_file, 'r') as f:
        items = json.load(f)
    image_paths, mask_paths = [], []
    for item in items:
        fn = item['filename']
        image_paths.append(os.path.join(dataset_path, "images_enhanced", fn + ".jpg"))
        mask_paths.append(os.path.join(dataset_path, "mask", fn + ".png"))
    return image_paths, mask_paths


def load_bts_mask_paths(dataset_path):
    image_paths, mask_paths = [], []
    image_dir = os.path.join(dataset_path, "image")
    mask_dir = os.path.join(dataset_path, "semantic_label")

    for fn in sorted(os.listdir(image_dir)):
        fno = os.path.splitext(fn)[0]
        image_paths.append(os.path.join(image_dir, fno + ".jpg"))
        mask_paths.append(os.path.join(mask_dir, fno + ".png"))
    return image_paths, mask_paths


def load_textseg_mask_paths(dataset_path):
    train_image_paths, train_mask_paths = [], []
    val_image_paths, val_mask_paths = [], []
    test_image_paths, test_mask_paths = [], []

    train_image_dir = os.path.join(dataset_path, "train_images")
    val_image_dir = os.path.join(dataset_path, "val_images")
    test_image_dir = os.path.join(dataset_path, "test_images")

    train_mask_dir = os.path.join(dataset_path, "train_gt")
    val_mask_dir = os.path.join(dataset_path, "val_gt")
    test_mask_dir = os.path.join(dataset_path, "test_gt")

    for fn in sorted(os.listdir(train_image_dir)):
        fno = os.path.splitext(fn)[0]
        train_image_paths.append(os.path.join(train_image_dir, fno + ".jpg"))
        train_mask_paths.append(os.path.join(train_mask_dir, fno + ".png"))

    for fn in sorted(os.listdir(val_image_dir)):
        fno = os.path.splitext(fn)[0]
        val_image_paths.append(os.path.join(val_image_dir, fno + ".jpg"))
        val_mask_paths.append(os.path.join(val_mask_dir, fno + ".png"))

    for fn in sorted(os.listdir(test_image_dir)):
        fno = os.path.splitext(fn)[0]
        test_image_paths.append(os.path.join(test_image_dir, fno + ".jpg"))
        test_mask_paths.append(os.path.join(test_mask_dir, fno + ".png"))

    return train_image_paths, train_mask_paths, val_image_paths, val_mask_paths, test_image_paths, test_mask_paths


class SyntheticDataset(Dataset):
    def __init__(self, large_image_path, ctr_dataset_path, ctr_json_file,
                 bts_dataset_path, textseg_dataset_path, output_size=(1024, 1024), get_pil_image=False):

        self.large_image_dataset = load_lsdir_paths(large_image_path)
        bts_train_dataset_path = os.path.join(bts_dataset_path, "TRAIN")
        bts_eval_dataset_path = os.path.join(bts_dataset_path, "VAL")

        ctr_image_paths, ctr_mask_paths = load_ctr_mask_paths(ctr_dataset_path, ctr_json_file)
        bts_train_image_paths, bts_train_mask_paths = load_bts_mask_paths(bts_train_dataset_path)
        bts_eval_image_paths, bts_eval_mask_paths = load_bts_mask_paths(bts_eval_dataset_path)
        (textseg_train_image_paths,
         textseg_train_mask_paths,
         textseg_val_image_paths,
         textseg_val_mask_paths,
         textseg_test_image_paths,
         textseg_test_mask_paths) = load_textseg_mask_paths(textseg_dataset_path)

        self.text_crop_paths = ctr_image_paths + bts_train_image_paths + bts_eval_image_paths + textseg_train_image_paths + textseg_val_image_paths + textseg_test_image_paths
        self.text_mask_paths = ctr_mask_paths + bts_train_mask_paths + bts_eval_mask_paths + textseg_train_mask_paths + textseg_val_mask_paths + textseg_test_mask_paths

        self.output_size = output_size
        self.transform = transforms.ToTensor()
        self.random_crop = transforms.RandomCrop(self.output_size)
        self.get_pil_image = get_pil_image

    def __len__(self):
        return len(self.large_image_dataset)

    def __getitem__(self, idx):
        # Load a large image from the dataset
        large_image = self.large_image_dataset[idx]
        if isinstance(large_image, str):
            if os.path.isdir(large_image):
                raise ValueError(f"Expected file path but got directory: {large_image}")
            large_image = Image.open(large_image)
        elif isinstance(large_image, torch.Tensor):
            large_image = transforms.ToPILImage()(large_image)

        # Random crop from the large image using torchvision's RandomCrop
        large_image = self.random_crop(large_image)
        large_w, large_h = large_image.size

        # Create a canvas from the large image
        canvas = large_image.copy()

        # Randomly select the number of crop images to paste (4 to 6)
        num_crops = random.randint(1, 5)
        selected_crops = random.sample(range(len(self.text_crop_paths)), num_crops)

        # Track occupied bounding boxes and annotations
        occupied_bboxes = []
        mask_full_image = Image.new('L', large_image.size, 0)

        for crop_idx in selected_crops:
            text_crop_path = self.text_crop_paths[crop_idx]
            text_mask_path = self.text_mask_paths[crop_idx]
            crop_image = Image.open(text_crop_path)
            mask_image = np.array(Image.open(text_mask_path))
            if mask_image.max() < 255:
                mask_image = (mask_image > 0).astype(np.uint8) * 255
            mask_image = Image.fromarray(mask_image)

            crop_w, crop_h = crop_image.size
            if crop_w > crop_h:
                new_w = large_w
                new_h = int(crop_h * (large_w / crop_w))
            else:
                new_h = large_h
                new_w = int(crop_w * (large_h / crop_h))

            crop_w, crop_h = new_w, new_h

            min_scale = 0.15
            scale = random.uniform(0.15, 1.0)
            placed = False
            while scale >= min_scale:
                scaled_w, scaled_h = int(crop_w * scale), int(crop_h * scale)
                # Attempt to place without overlap
                for _ in range(100):  # Retry up to 100 times
                    x = random.randint(0, large_w - scaled_w)
                    y = random.randint(0, large_h - scaled_h)
                    new_bbox = (x, y, x + scaled_w, y + scaled_h)

                    overlaps = any(self.check_overlap(new_bbox, existing_bbox) for existing_bbox in occupied_bboxes)
                    if not overlaps:
                        scaled_image = crop_image.resize((scaled_w, scaled_h), Image.LANCZOS)
                        scaled_mask = mask_image.resize((scaled_w, scaled_h), Image.LANCZOS)
                        canvas.paste(scaled_image, (x, y))
                        mask_full_image.paste(scaled_mask, (x, y))
                        occupied_bboxes.append(new_bbox)
                        placed = True
                        break

                # If placed successfully, move to the next crop
                if placed:
                    break

                # Reduce scale if not placed
                scale *= 0.9

            if not placed:
                print(f"Warning: Unable to place crop image at index {crop_idx} within bounds and scale constraints.")

        # Convert the final composed image to a tensor
        return {
            "image": self.transform(canvas),
            "mask": self.transform(mask_full_image)
        }

    @staticmethod
    def check_overlap(bbox1, bbox2):
        """Check if two bounding boxes overlap."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Overlap exists if one bbox is not completely to the left, right, above, or below the other
        return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)


def custom_collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        if key == 'anno':
            # Keep 'anno' as a list without stacking to avoid size mismatch error
            batch_dict[key] = [item[key] for item in batch]
        else:
            # Stack other keys (e.g., 'data') as usual
            batch_dict[key] = torch.stack([item[key] for item in batch])

    return batch_dict


if __name__ == '__main__':
    root_path = "[root_path]"
    gt_path = os.path.join(root_path, 'gt')
    lr_path = os.path.join(root_path, 'lr')
    sr_bicubic_path = os.path.join(root_path, 'sr_bicubic')
    mask_path = os.path.join(root_path, 'mask')
    os.makedirs(gt_path, exist_ok=True)
    # os.makedirs(lr_path, exist_ok=True)
    os.makedirs(sr_bicubic_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    epochs = 100
    batch_size = 1
    dataset_settings = get_dataset_settings()
    train_dataset = RealESRGANDataset(dataset_settings)
    train_dataloader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                       num_workers=16, drop_last=True, collate_fn=custom_collate_fn)
    degradation_settings = get_degradation_settings()
    step = 0
    bar = tqdm(range(len(train_dataset) * epochs))
    with torch.no_grad():
        for epoch in range(epochs):
            for num_batch, train_batch in enumerate(train_dataloader):
                lr_batch, gt_batch, mask_batch = realesrgan_degradation(train_batch,
                                                                        args_degradation=degradation_settings)
                sr_bicubic_batch = F.interpolate(lr_batch, size=(gt_batch.size(-2), gt_batch.size(-1)), mode='bicubic')

                for i in range(batch_size):
                    lr = lr_batch[i, ...]
                    gt = gt_batch[i, ...]
                    mask = mask_batch[i, ...]
                    sr_bicubic = sr_bicubic_batch[i, ...]

                    step += 1
                    bar.update(step)
                    bar.set_description('process {} images...'.format(step))

                    lr_save_path = os.path.join(lr_path, '{}.png'.format(str(step).zfill(7)))
                    gt_save_path = os.path.join(gt_path, '{}.png'.format(str(step).zfill(7)))
                    sr_bicubic_save_path = os.path.join(sr_bicubic_path, '{}.png'.format(str(step).zfill(7)))
                    mask_save_path = os.path.join(mask_path, '{}.png'.format(str(step).zfill(7)))

                    # saver.save_image(lr, save_path=lr_save_path)
                    saver.save_image(gt, save_path=gt_save_path)
                    saver.save_image(sr_bicubic, save_path=sr_bicubic_save_path)
                    saver.save_image(mask, save_path=mask_save_path)

                del lr_batch, gt_batch, sr_bicubic_batch, mask_batch
                torch.cuda.empty_cache()
