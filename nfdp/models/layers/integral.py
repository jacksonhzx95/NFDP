import torch
import torch.nn as nn
from torch.nn import functional as F


def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    accu_y = heatmaps.sum(dim=3)
    # accu_z = heatmaps.sum(dim=3)
    # accu_z = accu_z.sum(dim=3)
    accu_x = accu_x * torch.arange(float(x_dim)).to(accu_x.device)
    accu_y = accu_y * torch.arange(float(y_dim)).to(accu_y.device)
    # accu_z = accu_z * torch.arange(float(z_dim)).to(device)
    # accu_x = accu_x * \
    #          torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[
    #              0]
    # accu_y = accu_y * \
    #          torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[
    #              0]
    # accu_z = accu_z * \
    #          torch.cuda.comm.broadcast(torch.arange(z_dim).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[
    #              0]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    # accu_z = accu_z.sum(dim=2, keepdim=True)

    return accu_x, accu_y  # , accu_z


def generate_2d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    # accu_x = accu_x.sum(dim=2)
    # accu_y = heatmaps.sum(dim=2)
    accu_y = heatmaps.sum(dim=3)

    accu_x = accu_x * torch.arange(float(x_dim)).to(accu_x.device)
    accu_y = accu_y * torch.arange(float(y_dim)).to(accu_y.device)

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    return accu_x, accu_y


def softmax_integral_tensor(preds, num_joints, output_3d, hm_width, hm_height):
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = F.softmax(preds, 2)

    x, y = generate_2d_integral_preds_tensor(preds, num_joints, hm_width, hm_height)
    x = x / float(hm_width) - 0.5
    y = y / float(hm_height) - 0.5
    preds = torch.cat((x, y), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 2))
    return preds


'''
class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y
        '''


class Softmax_Integral(nn.Module):
    def __init__(self, num_joints, hm_width, hm_height):
        super(Softmax_Integral, self).__init__()
        self.num_joints = num_joints
        self.hm_width = hm_width
        self.hm_height = hm_height

    def forward(self, pred_hms):
        pred_hms = pred_hms.reshape((pred_hms.shape[0], self.num_joints, -1))
        pred_hms = F.softmax(pred_hms, 2)

        x, y = generate_2d_integral_preds_tensor(pred_hms, self.num_joints, self.hm_width, self.hm_height)
        x = x / float(self.hm_width) - 0.5
        y = y / float(self.hm_height) - 0.5
        preds = torch.cat((x, y), dim=2)
        preds = preds.reshape((pred_hms.shape[0], self.num_joints * 2))
        return preds
