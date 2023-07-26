import os
import torch.utils.data as data
# import pre_proc
import yaml
from easydict import EasyDict as edict
from nfdp.models import builder
import cv2
from scipy.io import loadmat
import numpy as np
from torch.utils.data import DataLoader
from nfdp.models.builder import DATASET
from nfdp.utils.presets import SimpleTransform, ScoliosisTransform


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


def load_gt_pts(annopath):
    pts = loadmat(annopath)['p2']  # num x 2 (x,y)
    pts = rearrange_pts(pts)
    return pts

'''def _check_load_keypoints(self, coco, entry):

            if self._check_centers and self._train:
                bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
                ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                if (num_vis / 80.0 + 47 / 80.0) > ks:
                    continue

'''


def scoliosis_pts_process(pts):
    joints_3d = np.zeros((len(pts), 2, 2), dtype=np.float32)
    for i in range(len(pts)):
        joints_3d[i, 0, 0] = pts[i][0]
        joints_3d[i, 1, 0] = pts[i][1]
        joints_3d[i, :2, 1] = 1
    return joints_3d


@DATASET.register_module
class Scoliosis_X_ray(data.Dataset):
    CLASSES = ['spine']
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                   60, 61, 62, 63, 64, 65, 66, 67]
    num_joints = 68
    joint_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
                   [10, 11], [12, 13], [14, 15], [16, 17], [18, 19],
                   [20, 21], [22, 23], [24, 25], [26, 27], [28, 29],
                   [30, 31], [32, 33], [34, 35], [36, 37], [38, 39],
                   [40, 41], [42, 43], [44, 45], [46, 47], [48, 49],
                   [50, 51], [52, 53], [54, 55], [56, 57], [58, 59],
                   [60, 61], [62, 63], [64, 65], [66, 67]
                   ]
    joints_name = (
        'T1_LU', 'T1_RU', 'T1_LD', 'T1_RD',
        'T2_LU', 'T2_RU', 'T2_LD', 'T2_RD',
        'T3_LU', 'T3_RU', 'T3_LD', 'T3_RD',
        'T4_LU', 'T4_RU', 'T4_LD', 'T4_RD',
        'T5_LU', 'T5_RU', 'T5_LD', 'T5_RD',
        'T6_LU', 'T6_RU', 'T6_LD', 'T6_RD',
        'T7_LU', 'T7_RU', 'T7_LD', 'T7_RD',
        'T8_LU', 'T8_RU', 'T8_LD', 'T8_RD',
        'T9_LU', 'T9_RU', 'T9_LD', 'T9_RD',
        'T10_LU', 'T10_RU', 'T10_LD', 'T10_RD',
        'T11_LU', 'T11_RU', 'T11_LD', 'T11_RD',
        'T12_LU', 'T12_RU', 'T12_LD', 'T12_RD',
        'L1_LU', 'L1_RU', 'L1_LD', 'L1_RD',
        'L2_LU', 'L2_RU', 'L2_LD', 'L2_RD',
        'L3_LU', 'L3_RU', 'L3_LD', 'L3_RD',
        'L4_LU', 'L4_RU', 'L4_LD', 'L4_RD',
        'L5_LU', 'L5_RU', 'L5_LD', 'L5_RD',
    )

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
        # self.label_dir =os.path.join(self._root, self._ann_file)
        if 'AUG' in cfg.keys():
            self._scale_factor = cfg['AUG']['SCALE_FACTOR']
            self._rot = cfg['AUG']['ROT_FACTOR']
            self.flip = cfg['AUG']['FLIP']
            self._shift = cfg['AUG']['SHIFT_FACTOR']
            self.num_joints_half_body = cfg['AUG']['NUM_JOINTS_HALF_BODY']
            self.prob_half_body = cfg['AUG']['PROB_HALF_BODY']
        else:
            self._scale_factor = 0
            self._rot = 0
            self.num_joints_half_body = -1
            self.prob_half_body = -1
            self._shift = (0, 0)
            self.flip = False

        self._input_size = self._preset_cfg['IMAGE_SIZE']
        self._output_size = self._preset_cfg['HEATMAP_SIZE']

        self._sigma = self._preset_cfg['SIGMA']

        self._check_centers = False

        self.num_class = len(self.CLASSES)
        self._loss_type = None
        # self._loss_type = cfg['heatmap2coord']

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                               10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47)
        self.lower_body_ids = (48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 66, 67)

        if self._preset_cfg['TYPE'] == 'simple':
            self.transformation = SimpleTransform(
                self, scale_factor=self._scale_factor,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=self._rot, sigma=self._sigma,
                train=self._train, loss_type=self._loss_type)
        elif self._preset_cfg['TYPE'] == 'scoliosis':
            self.transformation = ScoliosisTransform(
                self, scale_factor=self._scale_factor,
                input_size=self._input_size,
                output_size=self._output_size,
                flip=self.flip,
                rot=self._rot, sigma=self._sigma,
                train=self._train, loss_type=self._loss_type, shift=self._shift)
        else:
            raise NotImplementedError

        self.img_ids = sorted(os.listdir(self.img_dir))

    '''
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=4):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.class_name = ['__background__', 'cell']
        self.num_classes = 68
        self.img_dir = os.path.join(data_dir, 'data', self.phase)
        self.img_ids = sorted(os.listdir(self.img_dir))
    '''

    def load_image(self, index):
        image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index]))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.transpose(image, (2, 0, 1))
        return image

    def load_annoFolder(self, img_id):
        return os.path.join(self._ann_file, img_id + '.mat')

    def load_annotation(self, index):
        img_id = self.img_ids[index]
        annoFolder = self.load_annoFolder(img_id)
        pts = load_gt_pts(annoFolder)
        pts_3d = scoliosis_pts_process(pts)
        return pts_3d

    def __getitem__(self, index):
        '''
        img_path = self._items[idx]
        img_id = int(os.path.splitext(os.path.basename(img_path))[0])

        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox
        '''
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


if __name__ == '__main__':
    print('start')
    '''dataset = Scoliosis_X_ray(
        data_dir='/Users/huangzixun/Documents/datasets/scoliosis_keypoint(public)/boostnet_labeldata/',
        phase='training',
        input_h=1024,
        input_w=512)'''

    cfg_file_name = '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression/configs/512_res50_scoliosic_regress.yaml'
    cfg = update_config(cfg_file_name)
    # print(cfg)
    train_dataset = Scoliosis_X_ray(train=True,
                                    skip_empty=True,
                                    lazy_import=False,
                                    cfg=cfg)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0)
    train_loader = enumerate(train_loader)
    a = next(train_loader)
    print('Done')
