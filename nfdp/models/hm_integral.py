import torch
import torch.nn as nn
from easydict import EasyDict
from .layers.integral import Softmax_Integral
from .builder import NFDP
from .layers.Resnet import ResNet
from .layers.FPN_neck import FPN_neck_hm, FPNHead


def nets():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2),
                         nn.Tanh())


def nett():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))


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


@NFDP.register_module
class HeatmapIntegral(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(HeatmapIntegral, self).__init__()
        self._preset_cfg = cfg['PRESET']
        self.fc_dim = cfg['NUM_FC_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.height_dim = self._preset_cfg['IMAGE_SIZE'][0]
        self.width_dim = self._preset_cfg['IMAGE_SIZE'][1]
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm  # noqa: F401,F403
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")

        self.feature_channel = {
            18: 512,
            34: 512,
            50: 2048,
            101: 2048,
            152: 2048
        }[cfg['NUM_LAYERS']]
        self.decoder_feature_channel = {
            18: [64, 128, 256, 512],
            34: [64, 128, 256, 512],
            50: [256, 512, 1024, 2048],
            101: [256, 512, 1024, 2048],
            152: [256, 512, 1024, 2048],
        }[cfg['NUM_LAYERS']]

        self.neck = FPN_neck_hm(
            in_channels=self.decoder_feature_channel,
            out_channels=self.decoder_feature_channel[0],
            num_outs=4,
        )
        self.head = FPNHead(feature_strides=(4, 8, 16, 32),
                            in_channels=[self.decoder_feature_channel[0]] * 4,
                            channels=512,
                            num_classes=self.num_joints,
                            norm_cfg=dict(type='BN', requires_grad=True))

        self.hidden_list = cfg['HIDDEN_LIST']
        self.integral_hm = Softmax_Integral(num_pts=self.num_joints,
                                            hm_width=self.hm_width_dim,
                                            hm_height=self.hm_height_dim)


        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)


    def _initialize(self):
        pass

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]

        feats = self.preact.forward_feat(x)

        feats = self.neck(feats)

        output_hm = self.head(feats)
        out_coord = self.integral_hm(output_hm)


        # (B, N, 2)
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)
        scores = 1 - pred_pts

        scores = torch.mean(scores, dim=2, keepdim=True)

        output = EasyDict(
            pred_pts=pred_pts,
            maxvals=scores.float(),
        )

        return output
