import torch
import torch.nn as nn
from torch.nn import functional as F


def generate_2d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    accu_y = heatmaps.sum(dim=3)

    accu_x = accu_x * torch.arange(float(x_dim)).to(accu_x.device)
    accu_y = accu_y * torch.arange(float(y_dim)).to(accu_y.device)

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    return accu_x, accu_y


def softmax_integral_tensor(preds, num_joints, hm_width, hm_height):
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = F.softmax(preds, 2)

    x, y = generate_2d_integral_preds_tensor(preds, num_joints, hm_width, hm_height)
    x = x / float(hm_width) - 0.5
    y = y / float(hm_height) - 0.5
    preds = torch.cat((x, y), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 2))
    return preds




class Softmax_Integral(nn.Module):
    def __init__(self, num_pts, hm_width, hm_height):
        super(Softmax_Integral, self).__init__()
        self.num_pts = num_pts
        self.hm_width = hm_width
        self.hm_height = hm_height

    def forward(self, pred_hms):
        pred_hms = pred_hms.reshape((pred_hms.shape[0], self.num_pts, -1))
        pred_hms = F.softmax(pred_hms, 2)

        x, y = generate_2d_integral_preds_tensor(pred_hms, self.num_pts, self.hm_width, self.hm_height)
        x = x / float(self.hm_width) - 0.5
        y = y / float(self.hm_height) - 0.5
        preds = torch.cat((x, y), dim=2)
        preds = preds.reshape((pred_hms.shape[0], self.num_pts * 2))
        return preds
