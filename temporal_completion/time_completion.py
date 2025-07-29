from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import os, sys, copy, argparse
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
import pickle
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
import multiprocessing as mp

sys.path.insert(0, 'src')
import data_utils

MAX_SCENES = 850
TRAIN_REF_DIRPATH = os.path.join('training', 'nuscenes')
VAL_REF_DIRPATH = os.path.join('validation', 'nuscenes')
TEST_REF_DIRPATH = os.path.join('testing', 'nuscenes')

TRAIN_IMAGE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_image.txt')
TRAIN_LIDAR_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_lidar.txt')
TRAIN_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_ground_truth.txt')
TRAIN_GROUND_TRUTH_INTERP_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_ground_truth_interp.txt')

VAL_IMAGE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_image.txt')
VAL_LIDAR_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_lidar.txt')
VAL_GROUND_TRUTH_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_ground_truth.txt')
VAL_GROUND_TRUTH_INTERP_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_ground_truth_interp.txt')

VAL_IMAGE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_image-subset.txt')
VAL_LIDAR_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_lidar-subset.txt')
VAL_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_ground_truth-subset.txt')
VAL_GROUND_TRUTH_INTERP_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_ground_truth_interp-subset.txt')

parser = argparse.ArgumentParser()

parser.add_argument('--nuscenes_data_root_dirpath',
    type=str, required=True, help='Path to nuscenes dataset')
parser.add_argument('--nuscenes_data_derived_dirpath',
    type=str, required=True, help='Path to derived dataset')
parser.add_argument('--n_scenes_to_process',
    type=int, default=MAX_SCENES, help='Number of scenes to process')
parser.add_argument('--n_forward_frames_to_reproject',
    type=int, default=12, help='Number of forward frames to project onto a target frame')
parser.add_argument('--n_backward_frames_to_reproject',
    type=int, default=12, help='Number of backward frames to project onto a target frame')
parser.add_argument('--paths_only',
    action='store_true', help='If set, then only produce paths')
parser.add_argument('--n_thread',
    type=int, default=40, help='Number of threads to use in parallel pool')
parser.add_argument('--debug',
    action='store_true', help='If set, then enter debug mode')


args = parser.parse_args()


nusc = NuScenes(
    version='v1.0-trainval',
    dataroot=args.nuscenes_data_root_dirpath,
    verbose=True)

nusc_explorer = NuScenesExplorer(nusc)

def get_train_val_split_ids(debug=False):
    '''
    Given the nuscenes object, find out which scene ids correspond to which set.
    The split is taken from the official nuScene split available here:
    https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py

    Arg(s):
        debug : bool
            if set, then enter debug mode

    Returns:
        list[int] : list containing ids of the scenes that are training split
        list[int] : list containing ids of the scenes that are validation split
    '''

    train_file_name = os.path.join('data_split', 'train_ids.pkl')
    val_file_name = os.path.join('data_split', 'val_ids.pkl')

    open_file = open(train_file_name, "rb")
    train_ids = pickle.load(open_file)
    open_file.close()

    open_file = open(val_file_name, "rb")
    val_ids = pickle.load(open_file)
    open_file.close()

    if debug:
        train_ids_final = [1]
        return train_ids_final, val_ids

    return train_ids, val_ids

def point_cloud_to_image(nusc,
                         point_cloud,
                         lidar_sensor_token,
                         camera_token,
                         min_distance_from_camera=1.0):

    camera = nusc.get('sample_data', camera_token)
    lidar_sensor = nusc.get('sample_data', lidar_sensor_token)

    image_path = os.path.join(nusc.dataroot, camera['filename'])
    image = data_utils.load_image(image_path)
  
    pose_lidar_to_body = nusc.get('calibrated_sensor', lidar_sensor['calibrated_sensor_token'])
    point_cloud.rotate(Quaternion(pose_lidar_to_body['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_lidar_to_body['translation']))

    pose_body_to_global = nusc.get('ego_pose', lidar_sensor['ego_pose_token'])
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_body_to_global['translation']))

    pose_body_to_global = nusc.get('ego_pose', camera['ego_pose_token'])
    point_cloud.translate(-np.array(pose_body_to_global['translation']))
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix.T)

    pose_body_to_camera = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
    point_cloud.translate(-np.array(pose_body_to_camera['translation']))
    point_cloud.rotate(Quaternion(pose_body_to_camera['rotation']).rotation_matrix.T)

    depth = point_cloud.points[2, :]

    points = view_points(point_cloud.points[:3, :], np.array(pose_body_to_camera['camera_intrinsic']), normalize=True)

    mask = np.ones(depth.shape[0], dtype=bool)
    mask = np.logical_and(mask, depth > min_distance_from_camera)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < image.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < image.shape[0] - 1)

    points = points[:, mask]
    depth = depth[mask]

    return points, depth, image

def camera_to_lidar_frame(nusc,
                          point_cloud,
                          lidar_sensor_token,
                          camera_token):


    camera = nusc.get('sample_data', camera_token)
    lidar_sensor = nusc.get('sample_data', lidar_sensor_token)

    pose_camera_to_body = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
    point_cloud.rotate(Quaternion(pose_camera_to_body['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_camera_to_body['translation']))

    pose_body_to_global = nusc.get('ego_pose', camera['ego_pose_token'])
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_body_to_global['translation']))

    pose_body_to_global = nusc.get('ego_pose', lidar_sensor['ego_pose_token'])
    point_cloud.translate(-np.array(pose_body_to_global['translation']))
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix.T)

    pose_lidar_to_body = nusc.get('calibrated_sensor', lidar_sensor['calibrated_sensor_token'])
    point_cloud.translate(-np.array(pose_lidar_to_body['translation']))
    point_cloud.rotate(Quaternion(pose_lidar_to_body['rotation']).rotation_matrix.T)

    return point_cloud

def merge_lidar_point_clouds(nusc,
                             nusc_explorer,
                             current_sample_token,
                             n_forward,
                             n_backward):
    '''
    Merges Lidar point from multiple samples and adds them to a single depth image
    Picks current_sample_token as reference and projects lidar points from all other frames into current_sample.

    Arg(s):
        nusc : NuScenes Object
            nuScenes object instance
        nusc_explorer : NuScenesExplorer Object
            nuScenes explorer object instance
        current_sample_token : str
            token for accessing the current sample data
        n_forward : int
            number of frames to merge in the forward direction.
        n_backward : int
            number of frames to merge in the backward direction
    Returns:
        numpy[float32] : 2 x N of x, y for lidar points projected into the image
        numpy[float32] : N depths of lidar points

    '''

    current_sample = nusc.get('sample', current_sample_token)

    main_lidar_token = current_sample['data']['LIDAR_TOP']

    main_camera_token = current_sample['data']['CAM_FRONT']

    main_points_lidar, main_depth_lidar, main_image = nusc_explorer.map_pointcloud_to_image(
        pointsensor_token=main_lidar_token,
        camera_token=main_camera_token)

    main_image = np.asarray(main_image)

    main_lidar_image = np.zeros((main_image.shape[0], main_image.shape[1]))

    _, main_boxes, main_camera_intrinsic = nusc.get_sample_data(
        main_camera_token,
        box_vis_level=BoxVisibility.ANY,
        use_flat_vehicle_coordinates=False)

    main_points_lidar_quantized = np.round(main_points_lidar).astype(int)

    for point_idx in range(0, main_points_lidar_quantized.shape[1]):
        x = main_points_lidar_quantized[0, point_idx]
        y = main_points_lidar_quantized[1, point_idx]

        main_lidar_image[y, x] = main_depth_lidar[point_idx]

    main_validity_map = np.where(main_lidar_image > 0, 1, 0)

    n_forward_processed = 0
    n_backward_processed = 0

    next_sample = copy.deepcopy(current_sample)

    while next_sample['next'] != "" and n_forward_processed < n_forward:

        '''
        1. Load point cloud in `next' frame,
        2. Poject onto image to remove vehicle bounding boxes
        3. Backproject to camera frame
        '''

        next_sample_token = next_sample['next']
        next_sample = nusc.get('sample', next_sample_token)

        next_lidar_token = next_sample['data']['LIDAR_TOP']
        next_camera_token = next_sample['data']['CAM_FRONT']

        _, next_boxes, next_camera_intrinsics = nusc.get_sample_data(
            next_camera_token,
            box_vis_level=BoxVisibility.ANY,
            use_flat_vehicle_coordinates=False)

        next_points_lidar, next_depth_lidar, _ = nusc_explorer.map_pointcloud_to_image(
            pointsensor_token=next_lidar_token,
            camera_token=next_camera_token)

        next_lidar_image = np.zeros_like(main_lidar_image)

        next_points_lidar_quantized = np.round(next_points_lidar).astype(int)
        for idx in range(0, next_points_lidar_quantized.shape[-1]):
            x, y = next_points_lidar_quantized[0:2, idx]
            next_lidar_image[y, x] = next_depth_lidar[idx]
        for box in next_boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=next_camera_intrinsics, normalize=True)[:2, :]
                min_x = int(np.min(corners.T[:, 0]))
                min_y = int(np.min(corners.T[:, 1]))
                max_x = int(np.max(corners.T[:, 0]))
                max_y = int(np.max(corners.T[:, 1]))
                next_lidar_image[min_y:max_y, min_x:max_x] = 0
        next_lidar_points_y, next_lidar_points_x  = np.nonzero(next_lidar_image)
        next_lidar_points_z = next_lidar_image[next_lidar_points_y, next_lidar_points_x]
        x_y_homogeneous = np.stack([
            next_lidar_points_x,
            next_lidar_points_y,
            np.ones_like(next_lidar_points_x)],
            axis=0)

        x_y_lifted = np.matmul(np.linalg.inv(next_camera_intrinsics), x_y_homogeneous)
        x_y_z = x_y_lifted * np.expand_dims(next_lidar_points_z, axis=0)
        fake_intensity_array = np.ones(x_y_z.shape[1])
        fake_intensity_array = np.expand_dims(fake_intensity_array, axis=0)
        x_y_z = np.concatenate((x_y_z, fake_intensity_array), axis=0)
        next_point_cloud = LidarPointCloud(x_y_z)
        next_point_cloud = camera_to_lidar_frame(
            nusc=nusc,
            point_cloud=next_point_cloud,
            lidar_sensor_token=next_lidar_token,
            camera_token=next_camera_token)
        next_points_lidar_main, next_depth_lidar_main, _ = point_cloud_to_image(
            nusc=nusc,
            point_cloud=next_point_cloud,
            lidar_sensor_token=next_lidar_token,
            camera_token=main_camera_token,
            min_distance_from_camera=1.0)
        next_lidar_image_main = np.zeros_like(main_lidar_image)
        next_points_lidar_main_quantized = np.round(next_points_lidar_main).astype(int)

        for idx in range(0, next_points_lidar_main_quantized.shape[-1]):
            x, y = next_points_lidar_main_quantized[0:2, idx]
            next_lidar_image_main[y, x] = next_depth_lidar_main[idx]
        for box in main_boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=main_camera_intrinsic, normalize=True)[:2, :]
                min_x_main = int(np.min(corners.T[:, 0]))
                min_y_main = int(np.min(corners.T[:, 1]))
                max_x_main = int(np.max(corners.T[:, 0]))
                max_y_main = int(np.max(corners.T[:, 1]))
                next_lidar_image_main[min_y_main:max_y_main, min_x_main:max_x_main] = 0
        next_lidar_points_main_y, next_lidar_points_main_x  = np.nonzero(next_lidar_image_main)
        next_lidar_points_main_z = next_lidar_image_main[next_lidar_points_main_y, next_lidar_points_main_x]
        next_points_lidar_main = np.stack([
            next_lidar_points_main_x,
            next_lidar_points_main_y],
            axis=0)
        next_depth_lidar_main = next_lidar_points_main_z

        next_points_lidar_main_quantized = np.round(next_points_lidar_main).astype(int)

        for point_idx in range(0, next_points_lidar_main_quantized.shape[1]):
            x = next_points_lidar_main_quantized[0, point_idx]
            y = next_points_lidar_main_quantized[1, point_idx]

            is_not_occluded = \
                main_validity_map[y, x] == 1 and \
                next_depth_lidar_main[point_idx] < main_lidar_image[y, x]

            if is_not_occluded:
                main_lidar_image[y, x] = next_depth_lidar_main[point_idx]
            elif main_validity_map[y, x] != 1:
                main_lidar_image[y, x] = next_depth_lidar_main[point_idx]
                main_validity_map[y, x] = 1

        n_forward_processed = n_forward_processed + 1
    prev_sample = copy.deepcopy(current_sample)

    while prev_sample['prev'] != "" and n_backward_processed < n_backward:
        '''
        1. Load point cloud in `prev' frame,
        2. Poject onto image to remove vehicle bounding boxes
        3. Backproject to camera frame
        '''
        prev_sample_token = prev_sample['prev']
        prev_sample = nusc.get('sample', prev_sample_token)
        prev_lidar_token = prev_sample['data']['LIDAR_TOP']
        prev_camera_token = prev_sample['data']['CAM_FRONT']
        _, prev_boxes, prev_camera_intrinsics = nusc.get_sample_data(
            prev_camera_token,
            box_vis_level=BoxVisibility.ANY,
            use_flat_vehicle_coordinates=False)
        prev_points_lidar, prev_depth_lidar, _ = nusc_explorer.map_pointcloud_to_image(
            pointsensor_token=prev_lidar_token,
            camera_token=prev_camera_token)

        prev_lidar_image = np.zeros_like(main_lidar_image)
        prev_points_lidar_quantized = np.round(prev_points_lidar).astype(int)

        for idx in range(0, prev_points_lidar_quantized.shape[-1]):
            x, y = prev_points_lidar_quantized[0:2, idx]
            prev_lidar_image[y, x] = prev_depth_lidar[idx]
        for box in prev_boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=prev_camera_intrinsics, normalize=True)[:2, :]
                min_x = int(np.min(corners.T[:, 0]))
                min_y = int(np.min(corners.T[:, 1]))
                max_x = int(np.max(corners.T[:, 0]))
                max_y = int(np.max(corners.T[:, 1]))
                prev_lidar_image[min_y:max_y, min_x:max_x] = 0
        prev_lidar_points_y, prev_lidar_points_x  = np.nonzero(prev_lidar_image)
        prev_lidar_points_z = prev_lidar_image[prev_lidar_points_y, prev_lidar_points_x]
        x_y_homogeneous = np.stack([
            prev_lidar_points_x,
            prev_lidar_points_y,
            np.ones_like(prev_lidar_points_x)],
            axis=0)

        x_y_lifted = np.matmul(np.linalg.inv(prev_camera_intrinsics), x_y_homogeneous)
        x_y_z = x_y_lifted * np.expand_dims(prev_lidar_points_z, axis=0)
        fake_intensity_array = np.ones(x_y_z.shape[1])
        fake_intensity_array = np.expand_dims(fake_intensity_array, axis=0)
        x_y_z = np.concatenate((x_y_z, fake_intensity_array), axis=0)
        prev_point_cloud = LidarPointCloud(x_y_z)
        prev_point_cloud = camera_to_lidar_frame(
            nusc=nusc,
            point_cloud=prev_point_cloud,
            lidar_sensor_token=prev_lidar_token,
            camera_token=prev_camera_token)
        prev_points_lidar_main, prev_depth_lidar_main, _ = point_cloud_to_image(
            nusc=nusc,
            point_cloud=prev_point_cloud,
            lidar_sensor_token=prev_lidar_token,
            camera_token=main_camera_token,
            min_distance_from_camera=1.0)
        prev_lidar_image_main = np.zeros_like(main_lidar_image)
        prev_points_lidar_main_quantized = np.round(prev_points_lidar_main).astype(int)

        for idx in range(0, prev_points_lidar_main_quantized.shape[-1]):
            x, y = prev_points_lidar_main_quantized[0:2, idx]
            prev_lidar_image_main[y, x] = prev_depth_lidar_main[idx]
        for box in main_boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=main_camera_intrinsic, normalize=True)[:2, :]
                min_x_main = int(np.min(corners.T[:, 0]))
                min_y_main = int(np.min(corners.T[:, 1]))
                max_x_main = int(np.max(corners.T[:, 0]))
                max_y_main = int(np.max(corners.T[:, 1]))
                prev_lidar_image_main[min_y_main:max_y_main, min_x_main:max_x_main] = 0
        prev_lidar_points_main_y, prev_lidar_points_main_x  = np.nonzero(prev_lidar_image_main)
        prev_lidar_points_main_z = prev_lidar_image_main[prev_lidar_points_main_y, prev_lidar_points_main_x]
        prev_points_lidar_main = np.stack([
            prev_lidar_points_main_x,
            prev_lidar_points_main_y],
            axis=0)
        prev_depth_lidar_main = prev_lidar_points_main_z

        prev_points_lidar_main_quantized = np.round(prev_points_lidar_main).astype(int)

        for point_idx in range(0, prev_points_lidar_main_quantized.shape[1]):
            x = prev_points_lidar_main_quantized[0, point_idx]
            y = prev_points_lidar_main_quantized[1, point_idx]

            is_not_occluded = \
                main_validity_map[y, x] == 1 and \
                prev_depth_lidar_main[point_idx] < main_lidar_image[y, x]

            if is_not_occluded:
                main_lidar_image[y, x] = prev_depth_lidar_main[point_idx]
            elif main_validity_map[y, x] != 1:
                main_lidar_image[y, x] = prev_depth_lidar_main[point_idx]
                main_validity_map[y, x] = 1

        n_backward_processed = n_backward_processed + 1
    return_points_lidar_y, return_points_lidar_x = np.nonzero(main_lidar_image)
    return_depth_lidar = main_lidar_image[return_points_lidar_y, return_points_lidar_x]
    return_points_lidar = np.stack([
        return_points_lidar_x,
        return_points_lidar_y],
        axis=0)

    return return_points_lidar, return_depth_lidar

def lidar_depth_map_from_token(nusc,
                               nusc_explorer,
                               current_sample_token):

    current_sample = nusc.get('sample', current_sample_token)
    lidar_token = current_sample['data']['LIDAR_TOP']
    main_camera_token = current_sample['data']['CAM_FRONT']
    main_points_lidar, main_depth_lidar, main_image = nusc_explorer.map_pointcloud_to_image(
        pointsensor_token=lidar_token,
        camera_token=main_camera_token)

    depth_map = points_to_depth_map(main_points_lidar, main_depth_lidar, main_image)

    return depth_map

def points_to_depth_map(points, depth, image):
    image = np.asarray(image)
    depth_map = np.zeros((image.shape[0], image.shape[1]))

    points_quantized = np.round(points).astype(int)

    for pt_idx in range(0, points_quantized.shape[1]):
        x = points_quantized[0, pt_idx]
        y = points_quantized[1, pt_idx]
        depth_map[y, x] = depth[pt_idx]

    return depth_map

def process_scene(args):

    tag, \
        scene_id, \
        first_sample_token, \
        last_sample_token, \
        n_forward, \
        n_backward, \
        output_dirpath, \
        paths_only = args
    sample_id = 0
    sample_token = first_sample_token

    camera_image_paths = []
    lidar_paths = []
    ground_truth_paths = []
    ground_truth_interp_paths = []

    print('Processing scene_id={}'.format(scene_id))
    while sample_token != last_sample_token:
        current_sample = nusc.get('sample', sample_token)
        camera_token = current_sample['data']['CAM_FRONT']
        camera_sample = nusc.get('sample_data', camera_token)

        '''
        Set up paths
        '''
        camera_image_path = os.path.join(nusc.dataroot, camera_sample['filename'])

        dirpath, filename = os.path.split(camera_image_path)
        dirpath = dirpath.replace(nusc.dataroot, output_dirpath)
        filename = os.path.splitext(filename)[0]
        lidar_dirpath = dirpath.replace(
            'samples',
            os.path.join('lidar', 'scene_{}'.format(scene_id)))
        lidar_filename = filename + '.png'

        lidar_path = os.path.join(
            lidar_dirpath,
            lidar_filename)

        ground_truth_dirpath = dirpath.replace(
            'samples',
            os.path.join('ground_truth', 'scene_{}'.format(scene_id)))
        ground_truth_filename = filename + '.png'

        ground_truth_path = os.path.join(
            ground_truth_dirpath,
            ground_truth_filename)
        ground_truth_interp_dirpath = dirpath.replace(
            'samples',
            os.path.join('ground_truth_interp', 'scene_{}'.format(scene_id)))
        ground_truth_interp_filename = filename + '.png'

        ground_truth_interp_path = os.path.join(
            ground_truth_interp_dirpath,
            ground_truth_interp_filename)
        dirpaths = [
            lidar_dirpath,
            ground_truth_dirpath,
            ground_truth_interp_dirpath
        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                try:
                    os.makedirs(dirpath)
                except Exception:
                    pass

        '''
        Store file paths
        '''
        camera_image_paths.append(camera_image_path)
        lidar_paths.append(lidar_path)
        ground_truth_paths.append(ground_truth_path)
        ground_truth_interp_paths.append(ground_truth_interp_path)

        if not paths_only:

            '''
            Get camera data
            '''
            camera_image = data_utils.load_image(camera_image_path)

            '''
            Get lidar points projected to an image and save as PNG
            '''
            lidar_depth = lidar_depth_map_from_token(
                nusc=nusc,
                nusc_explorer=nusc_explorer,
                current_sample_token=sample_token)

            data_utils.save_depth(lidar_depth, lidar_path)
            points_lidar, depth_lidar = merge_lidar_point_clouds(
                nusc=nusc,
                nusc_explorer=nusc_explorer,
                current_sample_token=sample_token,
                n_forward=n_forward,
                n_backward=n_backward)

            '''
            Project point cloud onto the image plane and save as PNG
            '''
            ground_truth = points_to_depth_map(points_lidar, depth_lidar, camera_image)

            data_utils.save_depth(ground_truth, ground_truth_path)

            '''
            Interpolate missing points in ground truth point cloud and save as PNG
            '''
            validity_map = np.where(ground_truth > 0.0, 1.0, 0.0)
            ground_truth_interp = data_utils.interpolate_depth(
                ground_truth,
                validity_map)

            data_utils.save_depth(ground_truth_interp, ground_truth_interp_path)

        sample_id = sample_id + 1
        sample_token = current_sample['next']

    print('Finished {} samples in scene_id={}'.format(sample_id, scene_id))

    return (tag,
            camera_image_paths,
            lidar_paths,
            ground_truth_paths,
            ground_truth_interp_paths)



'''
Main function
'''
if __name__ == '__main__':

    use_multithread = args.n_thread > 1 and not args.debug

    pool_inputs = []
    pool_results = []

    train_camera_image_paths = []
    train_lidar_paths = []
    train_ground_truth_paths = []
    train_ground_truth_interp_paths = []

    val_camera_image_paths = []
    val_lidar_paths = []
    val_ground_truth_paths = []
    val_ground_truth_interp_paths = []

    train_ids, val_ids = get_train_val_split_ids()

    n_scenes_to_process = min(args.n_scenes_to_process, MAX_SCENES)
    n_train = len([s for s in range(n_scenes_to_process) if s in train_ids])
    n_val = len([s for s in range(n_scenes_to_process) if s in val_ids])
    print('Total Scenes to process: {}'.format(n_scenes_to_process))
    print('Training: {}  Validation: {}'.format(n_train, n_val))
    for scene_id in range(0, min(args.n_scenes_to_process, MAX_SCENES)):

        if scene_id in train_ids:
            tag = 'train'
        elif scene_id in val_ids:
            tag = 'val'
        else:
            raise ValueError('scene_id={} cannot be found in train or val split'.format(scene_id))

        current_scene = nusc.scene[scene_id]
        first_sample_token = current_scene['first_sample_token']
        last_sample_token = current_scene['last_sample_token']

        inputs = [
            tag,
            scene_id,
            first_sample_token,
            last_sample_token,
            args.n_forward_frames_to_reproject,
            args.n_backward_frames_to_reproject,
            args.nuscenes_data_derived_dirpath,
            args.paths_only
        ]

        pool_inputs.append(inputs)

        if not use_multithread:
            pool_results.append(process_scene(inputs))

    if use_multithread:
        with mp.Pool(args.n_thread) as pool:
            pool_results = pool.map(process_scene, pool_inputs)
    for results in pool_results:

        tag, \
            camera_image_scene_paths, \
            lidar_scene_paths, \
            ground_truth_scene_paths, \
            ground_truth_interp_scene_paths = results

        if tag == 'train':
            train_camera_image_paths.extend(camera_image_scene_paths)
            train_lidar_paths.extend(lidar_scene_paths)
            train_ground_truth_paths.extend(ground_truth_scene_paths)
            train_ground_truth_interp_paths.extend(ground_truth_interp_scene_paths)
        elif tag == 'val':
            val_camera_image_paths.extend(camera_image_scene_paths)
            val_lidar_paths.extend(lidar_scene_paths)
            val_ground_truth_paths.extend(ground_truth_scene_paths)
            val_ground_truth_interp_paths.extend(ground_truth_interp_scene_paths)
        else:
            raise ValueError('Found invalid tag: {}'.format(tag))
    val_camera_image_subset_paths = val_camera_image_paths[::2]
    val_lidar_subset_paths = val_lidar_paths[::2]
    val_ground_truth_subset_paths = val_ground_truth_paths[::2]
    val_ground_truth_interp_subset_paths = val_ground_truth_interp_paths[::2]
    outputs = [
        [
            'training',
            [
                [
                    'image',
                    train_camera_image_paths,
                    TRAIN_IMAGE_FILEPATH
                ], [
                    'lidar',
                    train_lidar_paths,
                    TRAIN_LIDAR_FILEPATH
                ], [
                    'ground truth',
                    train_ground_truth_paths,
                    TRAIN_GROUND_TRUTH_FILEPATH
                ], [
                    'interpolated ground truth',
                    train_ground_truth_interp_paths,
                    TRAIN_GROUND_TRUTH_INTERP_FILEPATH
                ]
            ]
        ], [
            'validation',
            [
                [
                    'image',
                    val_camera_image_paths,
                    VAL_IMAGE_FILEPATH
                ], [
                    'lidar',
                    val_lidar_paths,
                    VAL_LIDAR_FILEPATH
                ], [
                    'ground truth',
                    val_ground_truth_paths,
                    VAL_GROUND_TRUTH_FILEPATH
                ], [
                    'interpolated ground truth',
                    val_ground_truth_interp_paths,
                    VAL_GROUND_TRUTH_INTERP_FILEPATH
                ], [
                    'image subset',
                    val_camera_image_subset_paths,
                    VAL_IMAGE_SUBSET_FILEPATH
                ], [
                    'lidar subset',
                    val_lidar_subset_paths,
                    VAL_LIDAR_SUBSET_FILEPATH
                ], [
                    'ground truth subset',
                    val_ground_truth_subset_paths,
                    VAL_GROUND_TRUTH_SUBSET_FILEPATH
                ], [
                    'interpolated ground truth subset',
                    val_ground_truth_interp_subset_paths,
                    VAL_GROUND_TRUTH_INTERP_SUBSET_FILEPATH
                ]
            ]
        ]
    ]

    for dirpath in [TRAIN_REF_DIRPATH, VAL_REF_DIRPATH, TEST_REF_DIRPATH]:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    for output_info in outputs:

        tag, output = output_info
        for output_type, paths, filepath in output:

            print('Storing {} {} {} file paths into: {}'.format(
                len(paths), tag, output_type, filepath))
            data_utils.write_paths(filepath, paths)