import os
import csv
import torch.utils.data as data
import yaml
from easydict import EasyDict as edict
import cv2
from scipy.io import loadmat
import numpy as np
from nfdp.models.builder import DATASET
from nfdp.utils.presets import Transform


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config


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

def load_csv(file_name, num_landmarks, dim):
    landmarks_dict = {}
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            id = row[0]
            landmarks = []
            num_entries = dim * num_landmarks + 1
            assert num_entries == len(row), 'number of row entries ({}) and landmark coordinates ({}) do not match'.format(num_entries, len(row))
            for i in range(1, dim * num_landmarks + 1, dim):
                if dim == 2:
                    coords = np.array([float(row[i]), float(row[i + 1])], np.float32)
                elif dim == 3:
                    coords = np.array([float(row[i]), float(row[i + 1]), float(row[i + 2])], np.float32)

                landmarks.append(coords)
            landmarks = np.array(landmarks)
            landmarks_dict[id] = landmarks
    return landmarks_dict


def load_gt_pts(annopath):
    pts = loadmat(annopath)['p2']  # num x 2 (x,y)
    pts = rearrange_pts(pts)
    return pts



def pts_process(pts):
    joints_ed = np.zeros((len(pts), 2, 2), dtype=np.float32)
    for i in range(len(pts)):
        joints_ed[i, 0, 0] = pts[i][0]
        joints_ed[i, 1, 0] = pts[i][1]
        joints_ed[i, :2, 1] = 1
    return joints_ed


@DATASET.register_module
class Hand_X_ray(data.Dataset):
    CLASSES = ['Hand']

    def __init__(self,
                 train=True,
                 skip_empty=True,
                 lazy_import=False,
                 **cfg):
        self._cfg = cfg
        # cfg = cfg['cfg']['DATASET']['TRAIN']
        self._root = cfg['ROOT']
        self._img_prefix = cfg['IMG_PREFIX']
        self._ann_file = os.path.join(self._root, cfg['ANN'])
        self._preset_cfg = cfg['PRESET']
        self._lazy_import = lazy_import
        self._skip_empty = skip_empty
        self._train = train

        self.img_dir = os.path.join(self._root, self._img_prefix)
        self.landmarks_dict = load_csv(self._ann_file, self._preset_cfg['NUM_JOINTS'], dim=2)
        if 'AUG' in cfg.keys():
            self._scale_factor = cfg['AUG']['SCALE_FACTOR']
            self._rot = cfg['AUG']['ROT_FACTOR']
            self._shift = cfg['AUG']['SHIFT_FACTOR']
        else:
            self._scale_factor = 0
            self._rot = 0
            self._shift = (0, 0)

        # get image index from the split file
        # self.img_ids = []
        self.split_file = os.path.join(self._root, cfg['SPLIT'])
        self.img_ids = self.get_img_ids(self.split_file)

        self._input_size = self._preset_cfg['IMAGE_SIZE']
        self._output_size = self._preset_cfg['HEATMAP_SIZE']
        self._sigma = self._preset_cfg['SIGMA']

        self._check_centers = False


        self.num_class = len(self.CLASSES)
        # self._loss_type = None
        self._loss_type = cfg['heatmap2coord']

        self.transformation = Transform(
            self, scale_factor=self._scale_factor,
            input_size=self._input_size,
            output_size=self._output_size,
            rot=self._rot, sigma=self._sigma,
            train=self._train, loss_type=self._loss_type, shift=self._shift)



    def load_image(self, index):
        image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index] + '.jpg'))
        return image

    def get_img_ids(self, split):
        img_ids = []
        with open(split) as f:
            for line in f:
                img_name = line.strip()
                img_ids.append(img_name)
        return img_ids

    def load_annotation(self, index):

        img_id = self.img_ids[index]
        pts = self.landmarks_dict[img_id]
        pts_3d = pts_process(pts)
        return pts_3d

    def __getitem__(self, index):

        '''
        Parameters
        ----------
        index

        Returns
        -------

        '''

        img_id = self.img_ids[index]
        img = self.load_image(index)
        joints = self.load_annotation(index)
        label = dict(joints=joints)
        label['width'] = img.shape[1]
        label['height'] = img.shape[0]

        target = self.transformation(img, label)

        img = target.pop('image')
        return img, target, img_id

    def __len__(self):
        return len(self.img_ids)





