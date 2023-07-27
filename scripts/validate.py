"""Validation script."""
import logging
import os
import random
import sys
sys.path.insert(0, '/home/jackson/Documents/Project_BME/Python_code/NF/res-loglikelihood-regression')

import torch
import torch.multiprocessing as mp
from nfdp.models import builder
from nfdp.opt import cfg, opt
from nfdp.trainer import validate
from nfdp.utils.env import init_dist
from nfdp.utils.transforms import get_coord

num_gpu = torch.cuda.device_count()


def main():
    if opt.launcher in ['none', 'slurm']:
        main_worker(None, opt, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        opt.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(opt, cfg))


def main_worker(gpu, opt, cfg):
    if gpu is not None:
        opt.gpu = gpu

    init_dist(opt)

    torch.backends.cudnn.benchmark = True

    m = builder.build_nfdp(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print(f'Loading model from {opt.checkpoint}...')
    m.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'), strict=True)

    m.cuda(opt.gpu)
    m = torch.nn.parallel.DataParallel(m, device_ids=[opt.gpu])

    heatmap_to_coord = get_coord(cfg, cfg.DATA_PRESET.HEATMAP_SIZE)

    with torch.no_grad():

        overview, mean = validate(m, opt, cfg, heatmap_to_coord, opt.valid_batch)
        if cfg.DATA_PRESET.TYPE is 'spine':
            print(f'##### pe_mean: {mean}; MAPE: {overview}#####')
        elif cfg.DATA_PRESET.TYPE is 'cephalograms':
            print(overview)
            print(mean)
        elif cfg.DATA_PRESET.TYPE is 'hand':
            print(overview)
            print(mean)


if __name__ == "__main__":

    if opt.world_size > num_gpu:
        print(f'Wrong world size. Changing it from {opt.world_size} to {num_gpu}.')
        opt.world_size = num_gpu
    main()
