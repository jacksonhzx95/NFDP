# from .coco_det import Mscoco_det
from .custom import CustomDataset
from .ce_dataset import CE_X_ray
from .hand_dataset import Hand_X_ray
from .ce_dataset_fix import CE_X_ray_fix
# from .mscoco import Mscoco
# from .h36m import H36m
# from .h36m_mpii import H36mMpii
from .scoliosis_x_ray import Scoliosis_X_ray
__all__ = ['CustomDataset', 'Scoliosis_X_ray', 'CE_X_ray', 'Hand_X_ray', 'CE_X_ray_fix']
