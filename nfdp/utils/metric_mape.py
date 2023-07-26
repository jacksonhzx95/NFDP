import json
import cv2
import csv
import os
from scipy.io import loadmat
import numpy as np
from nfdp.utils.transforms import transform_preds, get_affine_transform, affine_transform
from nfdp.utils import cobb_evaluate
from nfdp.utils.util import get_center_scale
from nfdp.utils.landmark_statistics import LandmarkStatistics


def load_csv(file_name, num_landmarks, dim):
    landmarks_dict = {}
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            id = row[0]
            landmarks = []
            num_entries = dim * num_landmarks + 1
            assert num_entries == len(
                row), 'number of row entries ({}) and landmark coordinates ({}) do not match'.format(num_entries,
                                                                                                     len(row))
            # print(len(points_dict), name)
            for i in range(1, dim * num_landmarks + 1, dim):
                # print(i)
                if dim == 2:
                    coords = np.array([float(row[i]), float(row[i + 1])], np.float32)
                elif dim == 3:
                    coords = np.array([float(row[i]), float(row[i + 1]), float(row[i + 2])], np.float32)
                    # landmark = Landmark(coords)
                landmarks.append(coords)
            landmarks = np.array(landmarks)
            landmarks_dict[id] = landmarks
    return landmarks_dict


def rearrange_pts(pts):
    boxes = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k + 4, :]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:, 1])
        y_inds_r = np.argsort(pt_r[:, 1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
    return np.asarray(boxes, np.float32)


def cal_deo_ce(kpt_json, img_size):
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/face_landmark/'
    IMG_PREFIX = 'RawImage/ALL'
    ANN = '400_senior'
    ANN2 = '400_junior'
    gt_path = os.path.join(DATASET_PATH, ANN)
    gt_path2 = os.path.join(DATASET_PATH, ANN2)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    kpt_data = kpt_json
    kpt_h, kpt_w = img_size  # (512, 256) or (1024, 512)
    kpt_num = 19
    original_image_extend = [193.5, 240]
    image_size_in = [kpt_w, kpt_h]
    spacing = [float(np.max([es / (s / 1.25) for es, s in zip(original_image_extend, image_size_in)]))] * 2
    landmark_dist = []
    landmark_statistic = LandmarkStatistics()
    print(len(kpt_data))
    for i in range(len(kpt_data)):
        pred_pts = []
        gt_pts = []
        img_name = kpt_data[i]['image_id']
        kpt_coord = kpt_data[i]['keypoints']
        # load gt ann
        annoFolder = os.path.join(gt_path, img_name[:-4] + '.txt')
        annoFolder2 = os.path.join(gt_path2, img_name[:-4] + '.txt')
        pts1 = []
        pts2 = []
        with open(annoFolder, 'r') as f:
            lines = f.readlines()
            for i in range(kpt_num):
                coordinates = lines[i].replace('\n', '').split(',')
                coordinates_int = [int(i) for i in coordinates]
                pts1.append(coordinates_int)
        with open(annoFolder2, 'r') as f:
            lines = f.readlines()
            for i in range(kpt_num):
                coordinates = lines[i].replace('\n', '').split(',')
                coordinates_int = [int(i) for i in coordinates]
                pts2.append(coordinates_int)
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        gt_kpt = (pts1 + pts2) / 2

        # gt_kpt = np.array(pts)
        img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_COLOR)
        img_size = img.shape  # (H, W, C)

        img_h, img_w = img_size[:2]
        _aspect_ratio = kpt_w / kpt_h
        # need to check the scale_multi
        center, scale = get_center_scale(img_w, img_h, aspect_ratio=_aspect_ratio, scale_mult=1.25)
        trans = get_affine_transform(center, scale, 0, image_size_in)
        for j in range(kpt_num):
            coord_draw = np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])])
            gt_kpts = affine_transform(gt_kpt[j][0:2], trans)
            pred_pts.append((int(coord_draw[0]), int(coord_draw[1])))
            gt_pts.append((int(gt_kpts[0]), int(gt_kpts[1])))
            landmark_dist.append(np.sqrt((coord_draw[0] - gt_kpts[0]) ** 2 + (coord_draw[1] - gt_kpts[1]) ** 2))

        landmark_statistic.add_landmarks(image_id=img_name, predicted=pred_pts, groundtruth=gt_pts, spacing=spacing)
    overview_string = landmark_statistic.get_overview_string([2.0, 2.5, 3.0, 4.0])
    pe_mean, pe_std, pe_median = landmark_statistic.get_pe_statistics()

    return overview_string, pe_mean


def cal_deo_hand(kpt_json, img_size):
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/hand_x_ray/'
    IMG_PREFIX = 'Images'
    ANN = 'all.csv'
    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    kpt_data = kpt_json
    kpt_h, kpt_w = img_size  # (512, 256) or (1024, 512)
    original_image_extend = [193.5, 240]
    image_size_in = [512, 512]
    spacing = [float(np.max([es / s for es, s in zip(original_image_extend, image_size_in)]))] * 2
    kpt_num = 37
    gt_landmark = load_csv(gt_path, kpt_num, dim=2)
    landmark_dist = []
    landmark_statistic = LandmarkStatistics()
    print(len(kpt_data))
    for i in range(len(kpt_data)):

        pred_pts = []
        gt_pts = []
        img_name = kpt_data[i]['image_id']
        kpt_coord = kpt_data[i]['keypoints']
        # load gt ann
        gt_kpt = gt_landmark[img_name]

        img = cv2.imread(os.path.join(data_path, img_name + '.jpg'), cv2.IMREAD_COLOR)
        img_size = img.shape  # (H, W, C)

        img_h, img_w = img_size[:2]
        _aspect_ratio = kpt_w / kpt_h
        # need to check the scale_multi
        center, scale = get_center_scale(img_w, img_h, aspect_ratio=_aspect_ratio, scale_mult=1.25)
        trans = get_affine_transform(center, scale, 0, image_size_in)

        # center_beta = np.array([kpt_w_beta * 0.5, kpt_h_beta * 0.5])
        for j in range(kpt_num):
            coord_draw = np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])])
            gt_kpts = affine_transform(gt_kpt[j][0:2], trans)
            pred_pts.append((int(coord_draw[0]), int(coord_draw[1])))
            gt_pts.append((int(gt_kpts[0]), int(gt_kpts[1])))
            landmark_dist.append(np.sqrt((coord_draw[0] - gt_kpts[0]) ** 2 + (coord_draw[1] - gt_kpts[1]) ** 2))
        landmark_statistic.add_landmarks(image_id=img_name, predicted=pred_pts, groundtruth=gt_pts,
                                         normalization_factor=50, normalization_indizes=[1, 5])
    overview_string = landmark_statistic.get_overview_string([2.0, 4.0, 10.0])
    pe_mean, pe_std, pe_median = landmark_statistic.get_pe_statistics()
    return overview_string, pe_mean


def cal_mape(kpt_json, img_size):
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/scoliosis/xray/boostnet_labeldata/'
    IMG_PREFIX = 'data/test'
    ANN = 'labels/test'
    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    # if the kpt_json is str, run the following code
    # kpt_file = open(kpt_json)
    # kpt_json = json.load(kpt_file)
    kpt_data = kpt_json
    kpt_h, kpt_w = img_size  # (512, 256) or (1024, 512)
    kpt_num = 68
    pr_cobb_angles = []
    gt_cobb_angles = []
    landmark_dist = []
    print(len(kpt_data))
    for i in range(len(kpt_data)):
        pred_pts = []
        gt_pts = []
        img_name = kpt_data[i]['image_id']
        kpt_coord = kpt_data[i]['keypoints']
        gt_img_ann = loadmat(os.path.join(gt_path, img_name))['p2']
        gt_kpt = rearrange_pts(gt_img_ann)
        # read img
        img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_COLOR)
        img_size = img.shape  # (H, W, C)
        img_h, img_w = img_size[:2]
        _aspect_ratio = kpt_w / kpt_h
        center, scale = get_center_scale(img_w, img_h, aspect_ratio=_aspect_ratio, scale_mult=1.25)

        for j in range(kpt_num):
            coord_draw = transform_preds(np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])]), center, scale,
                                         [kpt_w, kpt_h])
            pred_pts.append((int(coord_draw[0]), int(coord_draw[1])))
            gt_pts.append((int(gt_kpt[j][0]), int(gt_kpt[j][1])))
            landmark_dist.append(np.sqrt((coord_draw[0] - gt_kpt[j][0]) ** 2 + (coord_draw[1] - gt_kpt[j][1]) ** 2))
        pr_cobb_angles.append(cobb_evaluate.cobb_angle_calc(pred_pts, img))
        gt_cobb_angles.append(cobb_evaluate.cobb_angle_calc(gt_pts, img))

    pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
    gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)
    out_abs = abs(gt_cobb_angles - pr_cobb_angles)
    out_add = gt_cobb_angles + pr_cobb_angles

    term1 = np.sum(out_abs, axis=1)
    term2 = np.sum(out_add, axis=1)
    mse = np.mean(landmark_dist)
    SMAPE = np.mean(term1 / term2 * 100)
    print('SMAPE1 is {}'.format(SMAPE_single_angle(gt_cobb_angles[:, 0], pr_cobb_angles[:, 0])))
    print('SMAPE2 is {}'.format(SMAPE_single_angle(gt_cobb_angles[:, 1], pr_cobb_angles[:, 1])))
    print('SMAPE3 is {}'.format(SMAPE_single_angle(gt_cobb_angles[:, 2], pr_cobb_angles[:, 2])))

    return mse, SMAPE


def SMAPE_single_angle(gt_cobb_angles, pr_cobb_angles):
    out_abs = abs(gt_cobb_angles - pr_cobb_angles)
    out_add = gt_cobb_angles + pr_cobb_angles

    term1 = out_abs
    term2 = out_add

    term2[term2 == 0] += 1e-5

    SMAPE = np.mean(term1 / term2 * 100)
    return SMAPE


def cal_mape_original(kpt_json, img_size):
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/scoliosis/xray/boostnet_labeldata/'
    IMG_PREFIX = 'data/test'
    ANN = 'labels/test'
    gt_path = os.path.join(DATASET_PATH, ANN)
    data_path = os.path.join(DATASET_PATH, IMG_PREFIX)
    kpt_file = open(kpt_json)
    kpt_data = json.load(kpt_file)
    kpt_h, kpt_w = img_size
    kpt_num = 68
    pr_cobb_angles = []
    gt_cobb_angles = []
    landmark_dist = []
    for i in range(len(kpt_data)):
        pred_pts = []
        gt_pts = []
        img_name = kpt_data[i]['image_id']
        kpt_coord = kpt_data[i]['keypoints']
        gt_img_ann = loadmat(os.path.join(gt_path, img_name))['p2']
        gt_kpt = rearrange_pts(gt_img_ann)
        # read img
        img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_COLOR)
        img_size = img.shape  # (H, W, C)
        img_h, img_w = img_size[:2]
        kpt_w_beta = kpt_w
        kpt_h_beta = kpt_h
        # print(img_size)
        # modify the exact w, h ratio
        if kpt_w > img_w / img_h * kpt_h:
            kpt_w_beta = img_w / img_h * kpt_h
        elif kpt_h > img_h / img_w * kpt_w:
            kpt_h_beta = img_h / img_w * kpt_w
        scale = np.array([img_w, img_h]) * 1.25
        center = np.array([img_w * 0.5, img_h * 0.5])
        scale_beta = np.array([kpt_w_beta, kpt_h_beta])
        center_beta = np.array([kpt_w_beta * 0.5, kpt_h_beta * 0.5])
        for j in range(kpt_num):
            coord_draw = transform_preds(np.array([int(kpt_coord[j * 3]), int(kpt_coord[j * 3 + 1])]), center, scale,
                                         [kpt_w, kpt_h])
            pred_pts.append((int(coord_draw[0]), int(coord_draw[1])))
            gt_pts.append((int(gt_kpt[j][0]), int(gt_kpt[j][1])))
            landmark_dist.append(abs(coord_draw[0] - gt_kpt[j][0] + (coord_draw[1] - gt_kpt[j][1])))
            landmark_dist.append(np.sqrt((coord_draw[0] - gt_kpt[j][0]) ** 2 + (coord_draw[1] - gt_kpt[j][1]) ** 2))
        pr_cobb_angles.append(cobb_evaluate.cobb_angle_calc(pred_pts, img))
        gt_cobb_angles.append(cobb_evaluate.cobb_angle_calc(gt_pts, img))

    pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
    gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)
    out_abs = abs(gt_cobb_angles - pr_cobb_angles)
    out_add = gt_cobb_angles + pr_cobb_angles

    term1 = np.sum(out_abs, axis=1)
    term2 = np.sum(out_add, axis=1)
    mse = np.mean(landmark_dist)
    SMAPE = np.mean(term1 / term2 * 100)
    return mse, SMAPE
