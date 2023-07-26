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
from nfdp.utils.presets import SimpleTransform, Transform


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


def spine_pts_process(pts):
    joints_ed = np.zeros((len(pts), 2, 2), dtype=np.float32)
    for i in range(len(pts)):
        joints_ed[i, 0, 0] = pts[i][0]
        joints_ed[i, 1, 0] = pts[i][1]
        joints_ed[i, :2, 1] = 1
    return joints_ed


@DATASET.register_module
class Spine_X_ray(data.Dataset):
    CLASSES = ['spine']
    num_joints = 68

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
            self._shift = cfg['AUG']['SHIFT_FACTOR']
        else:
            self._scale_factor = 0
            self._rot = 0
            self._shift = (0, 0)

        self._input_size = self._preset_cfg['IMAGE_SIZE']
        self._output_size = self._preset_cfg['HEATMAP_SIZE']

        self._sigma = self._preset_cfg['SIGMA']

        self._check_centers = False

        self.num_class = len(self.CLASSES)
        self._loss_type = None
        # self._loss_type = cfg['heatmap2coord']

        if self._preset_cfg['TYPE'] == 'spine':
            self.transformation = Transform(
                self, scale_factor=self._scale_factor,
                input_size=self._input_size,
                output_size=self._output_size,
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
        pts_3d = spine_pts_process(pts)
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
    train_dataset = Spine_X_ray(train=True,
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
