import argparse
import logging
import os
import random
import numpy as np
import torch
from PIL import Image
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
)
import cv2
import torch.nn as nn
import math
from transformers import CLIPTextModel, CLIPTokenizer
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_2d_condition_main import UNet2DConditionModel_main
from src.models.projection import My_proj
from transformers import CLIPVisionModelWithProjection
from inference.diffsusion_completion_pipeline import diffsusion_completion_pipeline
from utils.seed_all import seed_all
from utils.image_util import get_filled_for_latents
from tqdm.auto import tqdm


def load_and_process_mask(mask_path):
    image = Image.open(mask_path).convert('L')
    mask = np.array(image)
    mask = mask / 255.0
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return mask

def calculate_rmse_log(original_depth, predicted_depth, mask):
    """计算掩码区域内的对数RMSE (RMSE_log)"""
    # 掩码区域为255的像素点
    mask_indices = np.where(mask == 255)
    
    if len(mask_indices[0]) == 0:
        return 0.0  # 没有掩码区域，返回0
    
    original_values = original_depth[mask_indices]
    predicted_values = predicted_depth[mask_indices]
    
    # 过滤掉无效值
    valid_indices = np.where((original_values > 0) & (predicted_values > 0))
    if len(valid_indices[0]) == 0:
        return 0.0  # 没有有效的比较点
    
    original_valid = original_values[valid_indices]/256
    predicted_valid = predicted_values[valid_indices]/256
    
    # 计算对数差异
    log_diff = np.log(original_valid) - np.log(predicted_valid)
    rmse_log = np.sqrt(np.mean(np.square(log_diff)))
    
    return rmse_log

def create_random_mask(depth_img, mask_ratio=0.1):
    """创建随机遮罩，将深度图中随机遮挡10%的像素"""
    height, width = depth_img.shape
    
    # 只在有效深度值的区域生成随机掩码
    valid_depth_mask = (depth_img > 0)
    valid_indices = np.where(valid_depth_mask)
    num_valid_pixels = len(valid_indices[0])
    
    # 随机选择10%的有效像素点
    num_masked_pixels = int(num_valid_pixels * mask_ratio)
    random_indices = np.random.choice(num_valid_pixels, num_masked_pixels, replace=False)
    
    # 创建掩码
    mask = np.zeros_like(depth_img, dtype=np.uint8)
    mask[valid_indices[0][random_indices], valid_indices[1][random_indices]] = 255
    
    return mask


def calculate_rmse(original_depth, predicted_depth, mask):
    """计算掩码区域内的RMSE"""
    # 掩码区域为255的像素点
    mask_indices = np.where(mask == 255)
    
    if len(mask_indices[0]) == 0:
        return 0.0  # 没有掩码区域，返回0
    
    original_values = original_depth[mask_indices]
    predicted_values = predicted_depth[mask_indices]
    
    # 过滤掉无效值
    valid_indices = np.where((original_values > 0) & (predicted_values > 0))
    if len(valid_indices[0]) == 0:
        return 0.0  # 没有有效的比较点
    
    original_valid = original_values[valid_indices]
    predicted_valid = predicted_values[valid_indices]
    
    squared_diff = np.square(original_valid - predicted_valid)
    rmse = np.sqrt(np.mean(squared_diff))/256
    
    return rmse


def calculate_abs_rmse(original_depth, predicted_depth, mask):
    """计算掩码区域内的绝对RMSE (ABSRMSE)"""
    # 掩码区域为255的像素点
    mask_indices = np.where(mask == 255)
    
    if len(mask_indices[0]) == 0:
        return 0.0  # 没有掩码区域，返回0
    
    original_values = original_depth[mask_indices]
    predicted_values = predicted_depth[mask_indices]
    
    # 过滤掉无效值
    valid_indices = np.where((original_values > 0) & (predicted_values > 0))
    if len(valid_indices[0]) == 0:
        return 0.0  # 没有有效的比较点
    
    original_valid = original_values[valid_indices]
    predicted_valid = predicted_values[valid_indices]
    
    # 计算绝对差异
    abs_diff = np.abs(original_valid - predicted_valid)
    abs_rmse = np.sqrt(np.mean(np.square(abs_diff)))
    
    return abs_rmse


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run depth inpainting and evaluation on first 100 images."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,  # quantitative evaluation uses 50 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=0,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--normalize_scale",
        type=float,
        default=1,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--denoising_unet_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--mapping_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--reference_unet_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default='/root/autodl-tmp/diffsusion_completion/train',
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--train_gt_dir",
        type=str,
        default='/root/autodl-tmp/diffsusion_completion/train_gt',
        help="Directory containing known depth images.",
    )
    parser.add_argument(
        "--blend",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.1,
        help="Ratio of pixels to mask for evaluation (default: 10%).",
    )
    
    args = parser.parse_args()
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    processing_res = args.processing_res
    seed = args.seed
    mask_ratio = args.mask_ratio
    
    output_dir_color = os.path.join(output_dir, "depth_colored")
    output_dir_npy = os.path.join(output_dir, "depth_npy")
    output_dir_masked = os.path.join(output_dir, "depth_masked")
    output_dir_masks = os.path.join(output_dir, "masks")
    output_dir_eval = os.path.join(output_dir, "evaluation")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    os.makedirs(output_dir_masked, exist_ok=True)
    os.makedirs(output_dir_masks, exist_ok=True)
    os.makedirs(output_dir_eval, exist_ok=True)
    
    logging.info(f"output dir = {output_dir}")
    
    train_dir = args.train_dir
    train_gt_dir = args.train_gt_dir
    input_image_paths = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(('.png', '.jpg'))])
    known_depth_paths = sorted([os.path.join(train_gt_dir, f) for f in os.listdir(train_gt_dir) if f.endswith(('.png', '.jpg'))])
    
    assert len(input_image_paths) == len(known_depth_paths), "The number of input images and known depth images must be the same."
    
    # 限制处理前100张图片
    input_image_paths = input_image_paths[:min(100, len(input_image_paths))]
    known_depth_paths = known_depth_paths[:min(100, len(known_depth_paths))]
    
    print(f"arguments: {args}")
    print(f"Processing {len(input_image_paths)} images.")
    
    if seed is None:
        import time
        seed = int(time.time())
    seed_all(seed)

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Model --------------------
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                                     subfolder='vae')
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                     subfolder='text_encoder')
    denoising_unet = UNet2DConditionModel_main.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet",
                                                    in_channels=12, sample_size=96,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True)
    reference_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet",
                                                    in_channels=4, sample_size=96,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True)
    image_enc = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    mapping_layer=My_proj()

    mapping_layer.load_state_dict(
        torch.load(args.mapping_path, map_location="cpu"),
        strict=False,
        )
    mapping_device = torch.device("cuda")
    mapping_layer.to(mapping_device )
    reference_unet.load_state_dict(
                torch.load(args.reference_unet_path, map_location="cpu"),
        )
    denoising_unet.load_state_dict(
        torch.load(args.denoising_unet_path, map_location="cpu"),
        strict=False,
        )
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder='tokenizer')
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path,subfolder='scheduler')
    pipe = diffsusion_completion_pipeline(reference_unet=reference_unet,
                                       denoising_unet = denoising_unet,  
                                       mapping_layer=mapping_layer,
                                       vae=vae,
                                       text_encoder=text_encoder,
                                       tokenizer=tokenizer,
                                       image_enc=image_enc,
                                       scheduler=scheduler,
                                       ).to('cuda')
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    # -------------------- Inference and evaluation --------------------
    rmse_values = []
    rmse_log_values = []  # 存储绝对RMSE值
    
    with torch.no_grad():
        for i in tqdm(range(len(input_image_paths)), desc="Processing images"):
            input_image_path = input_image_paths[i]
            known_depth_path = known_depth_paths[i]

            # 设置保存路径
            rgb_name_base = os.path.splitext(os.path.basename(input_image_path))[0]
            pred_name_base = rgb_name_base
            
            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            colored_save_path = os.path.join(output_dir_color, f"{pred_name_base}_colored.png")
            masked_save_path = os.path.join(output_dir_masked, f"{pred_name_base}_masked.npy")
            mask_save_path = os.path.join(output_dir_masks, f"{pred_name_base}_mask.png")

            # 加载图像和深度图
            input_image = Image.open(input_image_path)
            image = Image.open(known_depth_path)
            original_depth = np.array(image)
            
            # 1. 创建随机掩码，去除10%的点
            random_mask = create_random_mask(original_depth, mask_ratio)
            
            # 2. 保存掩码
            cv2.imwrite(mask_save_path, random_mask)
            
            # 3. 创建带有去除点的深度图
            modified_depth = original_depth.copy()
            modified_depth[random_mask == 255] = 0  # 将随机选择的点置为0
            
            # 4. 保存修改后的深度图
            np.save(masked_save_path, modified_depth)
            
            # 5. 使用掩码作为待修复区域
            mask = np.where(modified_depth == 0, 255, 0).astype(np.uint8)
            
            # 6. 使用fill_for_latents进行初步填充（如果不是精细化模式）
            if args.refine is not True:
                modified_depth = get_filled_for_latents(mask, modified_depth)
            
            # 7. 使用管道进行修复
            pipe_out = pipe(
                input_image,
                denosing_steps=denoise_steps,
                processing_res=processing_res,
                match_input_res=True,
                batch_size=1,
                color_map="Spectral",
                show_progress_bar=True,
                depth_numpy_origin=modified_depth,
                mask_origin=mask,
                guidance_scale=1,
                normalize_scale=args.normalize_scale,
                strength=args.strength,
                blend=args.blend)

            # 8. 获取修复后的深度图
            depth_pred = pipe_out.depth_np
            
            # 9. 计算RMSE和ABSRMSE（仅在随机遮挡的像素点上）
            rmse = calculate_rmse(original_depth, depth_pred, random_mask)
            rmse_log = calculate_rmse_log(original_depth, depth_pred, random_mask)
            rmse_values.append(rmse)
            rmse_log_values.append(rmse_log)
            
            print(f"Image {i+1}, RMSE: {rmse:.4f}, rmse_log: {rmse_log:.4f}")
            
            # 10. 保存修复后的深度图
            if os.path.exists(colored_save_path):
                logging.warning(f"Existing file: '{colored_save_path}' will be overwritten")

            np.save(npy_save_path, depth_pred.astype(np.uint16))
            pipe_out.depth_colored.save(colored_save_path)
        
        # 计算并保存总体评估结果
        avg_rmse = np.mean(rmse_values)
        std_rmse = np.std(rmse_values)
        min_rmse = np.min(rmse_values)
        max_rmse = np.max(rmse_values)
        
        avg_abs_rmse = np.mean(rmse_log_values)
        std_abs_rmse = np.std(rmse_log_values)
        min_abs_rmse = np.min(rmse_log_values)
        max_abs_rmse = np.max(rmse_log_values)
        
        with open(os.path.join(output_dir_eval, "evaluation_results.txt"), "w") as f:
            f.write(f"Evaluation Results for {len(rmse_values)} images:\n")
            f.write(f"Average RMSE: {avg_rmse:.4f}\n")
            f.write(f"Standard Deviation: {std_rmse:.4f}\n")
            f.write(f"Min RMSE: {min_rmse:.4f}\n")
            f.write(f"Max RMSE: {max_rmse:.4f}\n\n")
            
            f.write(f"Average ABSRMSE: {avg_abs_rmse:.4f}\n")
            f.write(f"Standard Deviation (ABSRMSE): {std_abs_rmse:.4f}\n")
            f.write(f"Min ABSRMSE: {min_abs_rmse:.4f}\n")
            f.write(f"Max ABSRMSE: {max_abs_rmse:.4f}\n\n")
            
            f.write("Per-image metrics:\n")
            for i, (rmse, abs_rmse) in enumerate(zip(rmse_values, rmse_log_values)):
                f.write(f"Image {i+1}: RMSE={rmse:.4f}, ABSRMSE={abs_rmse:.4f}\n")
        
        print(f"\nEvaluation completed.")
        print(f"Average RMSE: {avg_rmse:.4f}")
        print(f"Average ABSRMSE: {avg_abs_rmse:.4f}")
        print(f"Results saved to {os.path.join(output_dir_eval, 'evaluation_results.txt')}")