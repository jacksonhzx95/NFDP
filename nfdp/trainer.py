import json
import os
import pickle as pk
import numpy as np
import torch
from torch.nn.utils import clip_grad
from tqdm import tqdm
from nfdp.models import builder
from nfdp.utils.metrics import DataLogger, calc_accuracy, calc_coord_accuracy
from nfdp.utils.metric_mape import cal_mape, cal_deo_ce, cal_deo_hand

def clip_gradient(optimizer, max_norm, norm_type):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            clip_grad.clip_grad_norm_(param, max_norm, norm_type)


def train(opt, cfg, train_loader, m, criterion, optimizer):
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    m.train()
    hm_shape = cfg.DATA_PRESET.get('HEATMAP_SIZE')
    grad_clip = cfg.TRAIN.get('GRAD_CLIP', False)

    if opt.log:
        train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, (inps, labels, _) in enumerate(train_loader):
        inps = inps.cuda()

        for k, _ in labels.items():
            if k == 'type':
                continue

            labels[k] = labels[k].cuda(opt.gpu)
        # print(len(labels))
        output = m(inps, labels)

        loss = criterion(output, labels)
        if cfg.TEST.get('HEATMAP2COORD') == 'heatmap':
            acc = calc_accuracy(output, labels)
        elif cfg.TEST.get('HEATMAP2COORD') == 'coord':
            acc = calc_coord_accuracy(output, labels, hm_shape)

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc, batch_size)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            clip_gradient(optimizer, grad_clip.MAX_NORM, grad_clip.NORM_TYPE)
        optimizer.step()

        opt.trainIters += 1

        if opt.log:
            # TQDM
            train_loader.set_description(
                'loss: {loss:.8f} | acc: {acc:.4f}'.format(
                    loss=loss_logger.avg,
                    acc=acc_logger.avg)
            )

    if opt.log:
        train_loader.close()

    return loss_logger.avg, acc_logger.avg


def validate(m, opt, cfg, heatmap_to_coord, batch_size=1):
    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False, heatmap2coord=cfg.TEST.HEATMAP2COORD)
    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False, sampler=gt_val_sampler)
    kpt_json = []
    m.eval()
    hm_shape = cfg.DATA_PRESET.get('HEATMAP_SIZE')
    acc_val_sum = 0
    val_count = 0
    if opt.log:
        gt_val_loader = tqdm(gt_val_loader, dynamic_ncols=True)

    for inps, labels, img_ids in gt_val_loader:
        inps = inps.cuda()
        for k, _ in labels.items():
            if k == 'type':
                continue

            labels[k] = labels[k].cuda(opt.gpu)
        output = m(inps)

        if cfg.TEST.get('HEATMAP2COORD') == 'heatmap':
            acc_val = calc_accuracy(output, labels)
        elif cfg.TEST.get('HEATMAP2COORD') == 'coord':
            acc_val = calc_coord_accuracy(output, labels, hm_shape)
        acc_val_sum += acc_val
        val_count += 1
        for i in range(inps.shape[0]):
            pose_coords, pose_scores = heatmap_to_coord(
                output, idx=i)
            keypoints = np.concatenate((pose_coords[0], pose_scores[0]), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['image_id'] = str(img_ids[i])
            data['score'] = float(np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints
            kpt_json.append(data)

    with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{opt.rank}.pkl'), 'wb') as fid:
        pk.dump(kpt_json, fid, pk.HIGHEST_PROTOCOL)
    torch.distributed.barrier()  # Make sure all JSON files are saved

    if opt.rank == 0:
        kpt_json_all = []
        for r in range(opt.world_size):
            with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            os.remove(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'))
            kpt_json_all += kpt_pred

        with open(os.path.join(opt.work_dir, 'test_gt_kpt.json'), 'w') as fid:
            json.dump(kpt_json_all, fid)

        if cfg.DATA_PRESET.TYPE == 'spine':
            pe_mean, mape = cal_mape(kpt_json_all, cfg.DATA_PRESET.get('IMAGE_SIZE'))
            return mape, pe_mean
        elif cfg.DATA_PRESET.TYPE == 'cephalograms':
            overview, pe_mean = cal_deo_ce(kpt_json_all, cfg.DATA_PRESET.get('IMAGE_SIZE'))
            return overview, pe_mean
        elif cfg.DATA_PRESET.TYPE == 'hand':
            overview, pe_mean = cal_deo_hand(kpt_json_all, cfg.DATA_PRESET.get('IMAGE_SIZE'))
            return overview, pe_mean
    else:
        return 0, 0







