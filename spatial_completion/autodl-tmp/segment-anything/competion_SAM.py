import cv2
import numpy as np
import time
import torch
from concurrent.futures import ProcessPoolExecutor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import multiprocessing


import random



from fvcore.nn import FlopCountAnalysis, parameter_count_table


def inverse_distance_weighting(points, values, grid_x, grid_y):
    interpolated_values = np.zeros(grid_x.shape)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            distances = np.sqrt((points[:, 0] - grid_x[i, j])**2 + (points[:, 1] - grid_y[i, j])**2)
            weights = 1 / (distances**3 + 1e-10) 
            interpolated_values[i, j] = np.sum(weights * values) / np.sum(weights)
    return interpolated_values






def calculate_metrics(original, predicted):
    mask = original > 0
    
    if not np.any(mask):
        return {"rmse": 0, "rmse_log": 0, "d1": 0, "d2": 0, "d3": 0, 
                "abs_rel": 0, "sq_rel": 0, "log10": 0, "silog": 0}
    
    original_valid = original[mask].astype(np.float32)
    predicted_valid = predicted[mask].astype(np.float32)
    diff = predicted_valid - original_valid
    squared_diff = diff ** 2
    rmse = np.sqrt(np.mean(squared_diff))
    diff_log = np.log(predicted_valid + 1e-6) - np.log(original_valid + 1e-6)
    rmse_log = np.sqrt(np.mean(diff_log ** 2))

    thresh = np.maximum(predicted_valid / original_valid, original_valid / predicted_valid)
    d1 = np.mean(thresh < 1.25)
    d2 = np.mean(thresh < 1.25 ** 2)
    d3 = np.mean(thresh < 1.25 ** 3)

    abs_rel = np.mean(np.abs(diff) / original_valid)
    sq_rel = np.mean(squared_diff / original_valid)

    log10 = np.mean(np.abs(np.log10(predicted_valid + 1e-6) - np.log10(original_valid + 1e-6)))

    silog = np.sqrt(np.mean(diff_log ** 2) - 0.5 * (np.mean(diff_log) ** 2))
    
    return {
        "rmse": rmse, 
        "rmse_log": rmse_log, 
        "d1": d1, 
        "d2": d2, 
        "d3": d3, 
        "abs_rel": abs_rel, 
        "sq_rel": sq_rel, 
        "log10": log10, 
        "silog": silog
    }


def interpolate_point_cloud(mask, point_cloud_image):
    interpolated_image = np.zeros_like(point_cloud_image)

    object_mask = mask['segmentation']

    points = np.column_stack(np.where(object_mask))
    values = point_cloud_image[object_mask]

    valid_indices = values > 0
    points = points[valid_indices]
    values = values[valid_indices]

    if points.shape[0] < 10:
        return interpolated_image 

    grid_x, grid_y = np.mgrid[0:object_mask.shape[0]:10, 0:object_mask.shape[1]:10]

    interp_start_time = time.time()

    interpolated_values = inverse_distance_weighting(points, values, grid_x, grid_y)
    interp_end_time = time.time()
    interp_time = interp_end_time - interp_start_time

    N = points.shape[0]
    Gh = grid_x.shape[0]
    Gw = grid_x.shape[1]
    total_flops = Gh * Gw * 13 * N

    gflops = (total_flops / interp_time) / 1e9 if interp_time > 0 else 0
    print(f"IDW : N={N}, Grid={Gh}x{Gw}, usetime={interp_time:.4f}s, GFLOPS={gflops:.2f}")


    interpolated_values_resized = cv2.resize(interpolated_values, (object_mask.shape[1], object_mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    interpolated_image[object_mask] = interpolated_values_resized[object_mask]

    return interpolated_image



def process_image(file_name, sam_checkpoint, gpu_id):
 
    device = torch.device(f'cuda:{gpu_id}')
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device) 
    mask_generator = SamAutomaticMaskGenerator(sam)
    image_path = f"/root/autodl-tmp/segment-anything/train/{file_name}"
    point_cloud_image_path = f"/root/autodl-tmp/segment-anything/train160/{file_name.replace('.jpg', '.png')}"
    output_path = f"/root/autodl-tmp/segment-anything/SAMandTime/{file_name.replace('.jpg', '.png')}"
    image = cv2.imread(image_path)
    if image is None:
        print(f"can't read: {image_path}")
        return None
    
    try:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).to(device)

        processed_input = sam.preprocess(input_tensor)

        batched_input = processed_input.unsqueeze(0)

        flops = FlopCountAnalysis(sam.image_encoder, batched_input).total()
        
   
        inference_start_time = time.time()
  
        masks = mask_generator.generate(image)
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time

        gflops = (flops / inference_time) / 1e9 if inference_time > 0 else 0
        print(f"parmeter: \n{parameter_count_table(sam)}")
        print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
        print(f"Mask gen time: {inference_time:.4f} 秒")
        print(f"GFLOPS: {gflops:.2f}")
    except Exception as e:
        masks = mask_generator.generate(image)
    point_cloud_image = cv2.imread(point_cloud_image_path, cv2.IMREAD_GRAYSCALE)
    if point_cloud_image is None:
        print(f"can't read point_cloud_image: {point_cloud_image_path}")
        return None, None
    
    print(f"point_cloud_image.max: {point_cloud_image.max()}")

    original_point_cloud = point_cloud_image.copy()

    non_zero_indices = np.where(point_cloud_image > 0)
    total_points = len(non_zero_indices[0])
    print(f"total_points: {total_points}")
    if total_points > 0:
        points_to_delete = int(total_points * 0)
        delete_indices = random.sample(range(total_points), points_to_delete)
        
        for idx in delete_indices:
            row = non_zero_indices[0][idx]
            col = non_zero_indices[1][idx]
            point_cloud_image[row, col] = 0
        
        print(f"points_to_delete{points_to_delete}:({points_to_delete/total_points:.1%})")
    
    interpolated_image = np.zeros_like(point_cloud_image)
    for mask in masks:
        interpolated_image += interpolate_point_cloud(mask, point_cloud_image)
    
    print(f"interpolated_image: {interpolated_image.max()}")


    metrics = calculate_metrics(original_point_cloud, interpolated_image)
    

    print(f"RMSE: {metrics['rmse']:.4f}, RMSE_log: {metrics['rmse_log']:.4f}")
    print(f" d1: {metrics['d1']:.4f}, d2: {metrics['d2']:.4f}, d3: {metrics['d3']:.4f}")
    print(f"abs_rel: {metrics['abs_rel']:.4f}, sq_rel: {metrics['sq_rel']:.4f}")
    print(f"log10: {metrics['log10']:.4f}, silog: {metrics['silog']:.4f}")
    

    cv2.imwrite(output_path, interpolated_image)
    
    #metrics_file = "/home/disk/tianr/ZZX/SAM/TestNuscenes/metrics.txt"
    metrics_file = "/root/autodl-tmp/segment-anything/metrics.txt"
    with open(metrics_file, "a") as f:
        f.write(f"{file_name}, ")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}, ")
        f.write("\n")
    
    print(f"image saved: {output_path}")

    del sam
    torch.cuda.empty_cache()

    return metrics




def main():
    multiprocessing.set_start_method('spawn')
    with open("/root/autodl-tmp/segment-anything/train_image_path.txt", "r") as file:
        file_names = file.readlines()
    file_names = [name.strip() for name in file_names]

    # 限制只处理前100张图片
    # total_files = len(file_names)
    # file_names = file_names[:100]
    # print(f"原始图片总数: {total_files}，限制处理前100张图片")

   # sam_checkpoint = "/home/disk/tianr/ZZX/SAM/sam_vit_b_01ec64.pth"
    sam_checkpoint = "/root/autodl-tmp/segment-anything/sam_vit_b_01ec64.pth"
    num_gpus = 1
    processes_per_gpu = 1
    batch_size = num_gpus * processes_per_gpu


    all_metrics = {
        "rmse": [], "rmse_log": [], "d1": [], "d2": [], "d3": [],
        "abs_rel": [], "sq_rel": [], "log10": [], "silog": []
    }

    for i in range(0, len(file_names), batch_size):
        start_time = time.time()
        batch = file_names[i:i + batch_size]
        batch_results = []
        
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for j, file_name in enumerate(batch):
                gpu_id = j % num_gpus
                futures.append(executor.submit(process_image, file_name, sam_checkpoint, gpu_id))
                
            for future in futures:
                metrics = future.result()
                if metrics:
                    batch_results.append(metrics)
        
        # 收集该批次的结果
        for metrics in batch_results:
            for name, value in metrics.items():
                all_metrics[name].append(value)
            
        end_time = time.time()
        print(f"处理批次图片共耗时间: {end_time - start_time} 秒")

    # 计算并输出所有图片的平均指标
    if all_metrics["rmse"]:
        print(f"\n处理完成！所有{len(all_metrics['rmse'])}张图片的平均指标:")
        
        # 将平均指标写入文件
        metrics_file = "/root/autodl-tmp/segment-anything/metrics.txt"
        with open(metrics_file, "a") as f:
            f.write(f"\n平均指标 (共{len(all_metrics['rmse'])}张图片):\n")
            
            # 计算每个指标的平均值并输出
            for name, values in all_metrics.items():
                avg_value = sum(values) / len(values)
                print(f"平均{name}: {avg_value:.4f}")
                f.write(f"平均{name}: {avg_value:.4f}\n")
    else:
        print("没有成功处理任何图片，无法计算平均指标。")    

if __name__ == "__main__":
    main()