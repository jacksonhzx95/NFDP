import os
import torch.utils.data as data
import yaml
from easydict import EasyDict as edict
import cv2
from scipy.io import loadmat
import numpy as np
from nfdp.models.builder import DATASET
from nfdp.utils.presets import SimpleTransform, ScoliosisTransform, CETransform
import albumentations as A

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
class CE_X_ray(data.Dataset):
    CLASSES = ['CE']
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18]
    num_joints = 19
    joint_pairs = [[0, 0], [2, 2], [4, 4], [6, 6], [8, 8],
                   [10, 10], [12, 12], [14, 14], [16, 16], [18, 18]
                   ]
    joints_name = [
        'L0', 'L1', 'L2', 'L3',
        'L4', 'L5', 'L6', 'L7',
        'L8', 'L9', 'L10', 'L11',
        'L12', 'L13', 'L14', 'L15',
        'L16', 'L17', 'L18'
    ]

    def __init__(self,
                 train=True,
                 skip_empty=True,
                 lazy_import=False,
                 **cfg):
        self._cfg = cfg
        # cfg = cfg['cfg']['DATASET']['TRAIN']
        self._root = cfg['ROOT']
        self._img_prefix = cfg['IMG_PREFIX']
        self._ann_file = os.path.join(self._root, cfg['ANN'][0])
        self._ann_file2 = os.path.join(self._root, cfg['ANN'][1])
        self._preset_cfg = cfg['PRESET']
        self._lazy_import = lazy_import
        self._skip_empty = skip_empty
        self._train = train
        self.img_dir = os.path.join(self._root, self._img_prefix)

        if 'AUG' in cfg.keys():
            self._flip = cfg['AUG']['FLIP']
            self._scale_factor = cfg['AUG']['SCALE_FACTOR']
            self._rot = cfg['AUG']['ROT_FACTOR']
            self._shift = cfg['AUG']['SHIFT_FACTOR']
            self.num_joints_half_body = cfg['AUG']['NUM_JOINTS_HALF_BODY']
            self.prob_half_body = cfg['AUG']['PROB_HALF_BODY']
        else:
            self._flip = 0
            self._scale_factor = 0
            self._rot = 0
            self.num_joints_half_body = -1
            self.prob_half_body = -1
            self._shift = (0, 0)

        self._input_size = self._preset_cfg['IMAGE_SIZE']
        self._output_size = self._preset_cfg['HEATMAP_SIZE']

        self._sigma = self._preset_cfg['SIGMA']

        self._check_centers = False

        self.num_class = len(self.CLASSES)
        # self._loss_type = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                               10, 11, 12,)
        self.lower_body_ids = (13, 14, 15, 16, 17, 18,)
        self._loss_type = cfg['heatmap2coord']
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
                flip=self._flip,
                rot=self._rot, sigma=self._sigma,
                train=self._train, loss_type=self._loss_type, shift=self._shift)
        elif self._preset_cfg['TYPE'] == 'cephalograms':
            self.transformation = CETransform(
                self, scale_factor=self._scale_factor,
                input_size=self._input_size,
                output_size=self._output_size,
                flip=self._flip,
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
        return image

    def load_annoFolder(self, img_id):
        return os.path.join(self._ann_file, img_id[:-4] + '.txt'), os.path.join(self._ann_file2, img_id[:-4] + '.txt')

    def load_annotation(self, index):

        img_id = self.img_ids[index]
        annoFolder1, annoFolder2 = self.load_annoFolder(img_id)
        pts1 = []
        pts2 = []
        with open(annoFolder1, 'r') as f:
            lines = f.readlines()
            for i in range(self.num_joints):
                coordinates = lines[i].replace('\n', '').split(',')
                coordinates_int = [int(i) for i in coordinates]
                pts1.append(coordinates_int)
        with open(annoFolder2, 'r') as f:
            lines = f.readlines()
            for i in range(self.num_joints):
                coordinates = lines[i].replace('\n', '').split(',')
                coordinates_int = [int(i) for i in coordinates]
                pts2.append(coordinates_int)
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        pts = (pts1 + pts2) / 2
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
        # target['label']

        target = self.transformation(img, label)

        img = target.pop('image')

        return img, target, img_id

    def __len__(self):
        return len(self.img_ids)


# import numpy as np

if __name__ == '__main__':
    DATASET_PATH = '/home/jackson/Documents/Project_BME/Datasets/face_landmark/'
    content = open(os.path.join(DATASET_PATH, 'mini_train_input.bin'), 'rb').read()
    imgs = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
    content = open(os.path.join(DATASET_PATH, 'mini_train_gt.bin'), 'rb').read()
    gts = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))

    # spilit to train/val
    np.random.seed(2333)
    indices = np.random.permutation(imgs.shape[0])
    train_idx, val_idx = indices[500:], indices[:500]
    np.random.seed(None)
    train_img = imgs[train_idx, :, :]
    train_gt = gts[train_idx, :, :]
    val_img = imgs[val_idx, :, :]
    val_gt = gts[val_idx, :, :]

    fout = open('dataset/burst_raw/mini_train_input.bin', 'wb')
    fout.write(train_img.tobytes())
    fout = open('dataset/burst_raw/mini_train_gt.bin', 'wb')
    fout.write(train_gt.tobytes())

    fout = open('dataset/burst_raw/mini_val_input.bin', 'wb')
    fout.write(val_img.tobytes())
    fout = open('dataset/burst_raw/mini_val_gt.bin', 'wb')
    fout.write(val_gt.tobytes())


