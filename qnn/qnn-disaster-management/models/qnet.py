import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import Resnet18
from qnn.QN import QuaternionNorm2d as QN
from qnn.quaternion_layers import QuaternionConv

up_kwargs = {"mode": "bilinear", "align_corners": True}


class BatchNorm2d(QN):
    def __init__(self, num_features, activation="none"):
        super(BatchNorm2d, self).__init__(num_features=num_features)
        if activation == "leaky_relu":
            self.activation = nn.ReLU()
        elif activation == "none":
            self.activation = lambda x: x
        else:
            raise Exception("Accepted activation: ['leaky_relu']")

    def forward(self, x):
        return self.activation(super(BatchNorm2d, self).forward(x))
    
class FANet(nn.Module):
    def __init__(self, nclass=14, backbone="resnet18",
                 norm_layer=BatchNorm2d):
        super(FANet, self).__init__()
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass = nclass
        self.expansion = 1
        self.resnet = Resnet18(norm_layer=norm_layer)

        self.ffm_32 = LAFeatureFusionModule(
            512 * self.expansion, 256, 128, norm_layer=norm_layer
        )
        self.ffm_16 = LAFeatureFusionModule(
            256 * self.expansion, 256, 128, norm_layer=norm_layer
        )
        self.ffm_8 = LAFeatureFusionModule(
            128 * self.expansion, 256, 128, norm_layer=norm_layer
        )
        self.ffm_4 = LAFeatureFusionModule(
            64 * self.expansion, 256, 128, norm_layer=norm_layer
        )

        self.clslayer_32 = FPNOutput(128, 64, nclass, norm_layer=norm_layer)
        self.clslayer_16 = FPNOutput(128, 64, nclass, norm_layer=norm_layer)
        self.clslayer_8 = FPNOutput(256, 256, nclass, norm_layer=norm_layer)

    def forward(
        self,
        x,
    ):
        _, _, h, w = x.size()


        feat4, feat8, feat16, feat32 = self.resnet(x)

        upfeat_32, smfeat_32 = self.ffm_32(feat32, None, True, True)
        upfeat_16, smfeat_16 = self.ffm_16(feat16, upfeat_32, True, True)
        upfeat_8 = self.ffm_8(feat8, upfeat_16, True, False)
        smfeat_4 = self.ffm_4(feat4, upfeat_8, False, True)

        x = self._upsample_cat(smfeat_16, smfeat_4)

        x = self.clslayer_8(x)


        return x

    def _upsample_cat(self, x1, x2):
        _, _, H, W = x2.size()
        x1 = F.interpolate(x1, (H, W), **self._up_kwargs)
        x = torch.cat([x1, x2], dim=1)
        return x


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        ks=3,
        stride=1,
        padding=1,
        norm_layer=None,
        activation="leaky_relu",
        *args,
        **kwargs
    ):
        super(ConvBNReLU, self).__init__()


        self.conv = QuaternionConv(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False,
        )
        
        self.norm_layer = norm_layer
        if self.norm_layer is not None:
            self.bn = norm_layer(out_chan, activation=activation)
        else:
            self.bn = lambda x: x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class FPNOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, norm_layer=None, *args, **kwargs):
        super(FPNOutput, self).__init__()
        self.norm_layer = norm_layer
        self.conv = ConvBNReLU(
            in_chan, mid_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer,
        )
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        #self.conv_out = QuaternionConv(mid_chan, n_classes, kernel_size=1, stride=1, bias=False)
        # destruction expected 1x14x64x64, out 1x12x64x64  --> expected 4 or 3 dims input


    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class LAFeatureFusionModule(nn.Module):
    def __init__(
        self, in_chan, mid_chn=256, out_chan=128, norm_layer=None, *args, **kwargs
    ):
        super(LAFeatureFusionModule, self).__init__()
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        mid_chn = int(in_chan / 2)
        self.w_qs = ConvBNReLU(
            in_chan,
            32,
            ks=1,
            stride=1,
            padding=0,
            norm_layer=norm_layer,
            activation="none",
        )

        self.w_ks = ConvBNReLU(
            in_chan,
            32,
            ks=1,
            stride=1,
            padding=0,
            norm_layer=norm_layer,
            activation="none",
        )

        self.w_vs = ConvBNReLU(
            in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer
        )

        self.latlayer3 = ConvBNReLU(
            in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer
        )

        self.up = ConvBNReLU(
            in_chan, mid_chn, ks=1, stride=1, padding=1, norm_layer=norm_layer
        )
        self.smooth = ConvBNReLU(
            in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer
        )

    def forward(self, feat, up_fea_in, up_flag, smf_flag):
        query = self.w_qs(feat)
        key = self.w_ks(feat)
        value = self.w_vs(feat)

        N, C, H, W = feat.size()

        query_ = query.view(N, 32, -1).permute(0, 2, 1)
        query = F.normalize(query_, p=2, dim=2, eps=1e-12)

        key_ = key.view(N, 32, -1)
        key = F.normalize(key_, p=2, dim=1, eps=1e-12)

        value = value.view(N, C, -1).permute(0, 2, 1)

        f = key.matmul(value)
        y = query.matmul(f)
        y = y.permute(0, 2, 1).contiguous()

        y = y.view(N, C, H, W)
        W_y = self.latlayer3(y)
        p_feat = W_y + feat

        if up_flag and smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            up_feat = self.up(p_feat)
            smooth_feat = self.smooth(p_feat)
            return up_feat, smooth_feat

        if up_flag and not smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            up_feat = self.up(p_feat)
            return up_feat

        if not up_flag and smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            smooth_feat = self.smooth(p_feat)
            return smooth_feat

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, (H, W), **self._up_kwargs) + y






        

        
        
    
