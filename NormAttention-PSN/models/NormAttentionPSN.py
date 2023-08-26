import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
from torchvision.models import *
from torch.nn import *
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.conv6 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.conv7 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)  # from c2
        self.conv8 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.conv9 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.conv10 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.conv11 = model_utils.conv(batchNorm, 128, 256, k=3, stride=2, pad=1)  # from c8
        self.conv12 = model_utils.conv(batchNorm, 256, 256, k=3, stride=1, pad=1)
        self.conv13 = model_utils.conv(batchNorm, 256, 256, k=3, stride=1, pad=1)
        self.conv3down1 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)  # to c8
        self.conv4down1 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)  # to c9
        self.conv4down2 = model_utils.conv(batchNorm, 64, 256, k=3, stride=4, pad=1)  # to c11
        self.conv5down1 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)  # to c10
        self.conv5down2 = model_utils.conv(batchNorm, 64, 256, k=3, stride=4, pad=1)  # to c12
        self.conv10down1 = model_utils.conv(batchNorm, 128, 256, k=3, stride=2, pad=1)  # to c13
        self.convnor8 = model_utils.conv(batchNorm, 128, 64, k=1, stride=1, pad=0)
        self.convnor122 = model_utils.conv(batchNorm, 256, 128, k=1, stride=1, pad=0)
        self.convnor124 = model_utils.conv(batchNorm, 256, 64, k=1, stride=1, pad=0)
        self.convnor10 = model_utils.conv(batchNorm, 128, 64, k=1, stride=1, pad=0)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out3d = self.conv3down1(out3)
        out7 = self.conv7(out2)
        out8 = self.conv8(out7)
        out8up = torch.nn.functional.upsample(out8, scale_factor=2, mode='bilinear', align_corners=True)
        out8upnor = self.convnor8(out8up)
        out8 = torch.add(out8, out3d)
        out4 = self.conv4(out3)
        out4 = torch.add(out4, out8upnor)
        out4d1 = self.conv4down1(out4)
        out4d2 = self.conv4down2(out4)
        out9 = self.conv9(out8)
        out9 = torch.add(out9, out4d1)
        out5 = self.conv5(out4)
        out5d1 = self.conv5down1(out5)
        out5d2 = self.conv5down2(out5)
        out11 = self.conv11(out8)
        out11 = torch.add(out11, out4d2)
        out12 = self.conv12(out11)
        out12 = torch.add(out12, out5d2)
        out12up2 = torch.nn.functional.upsample(out12, scale_factor=2, mode='bilinear', align_corners=True)
        out12up2nor = self.convnor122(out12up2)
        out12up4 = torch.nn.functional.upsample(out12, scale_factor=4, mode='bilinear', align_corners=True)
        out12up4nor = self.convnor124(out12up4)
        out10 = self.conv10(out9)
        out10 = torch.add(out10, out5d1)
        out10 = torch.add(out10, out12up2nor)
        out10d = self.conv10down1(out10)
        out10up = torch.nn.functional.upsample(out10, scale_factor=2, mode='bilinear', align_corners=True)
        out10upnor = self.convnor10(out10up)
        out13 = self.conv13(out12)
        out13 = torch.add(out13, out10d)
        out6 = self.conv6(out5)
        out6 = torch.add(out6, out12up4nor)
        out6 = torch.add(out6, out10upnor)
        out6m = out6
        out10m = out10
        out13m = out13
        n6, c6, h6, w6 = out6m.data.shape
        out6m = out6m.view(-1)
        n10, c10, h10, w10 = out10m.data.shape
        out10m = out10m.view(-1)
        n13, c13, h13, w13 = out13m.data.shape
        out13m = out13m.view(-1)
        return out6m, [n6, c6, h6, w6], out10m, [n10, c10, h10, w10], out13m, [n13, c13, h13, w13]


class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.other = other
        self.deconv1 = model_utils.deconv(256, 128)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.deconv4 = model_utils.deconv(128, 64)
        self.deconv5 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.deconv6 = model_utils.conv(batchNorm, 192, 64, k=3, stride=1, pad=1)
        self.est_normal = self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x6, shape6, x10, shape10, x13, shape13):
        x6 = x6.view(shape6[0], shape6[1], shape6[2], shape6[3])
        x10 = x10.view(shape10[0], shape10[1], shape10[2], shape10[3])
        x13 = x13.view(shape13[0], shape13[1], shape13[2], shape13[3])
        out1 = self.deconv1(x13)
        out2 = self.deconv2(out1)
        out3 = self.deconv3(out2)
        out4 = self.deconv4(x10)
        out5 = self.deconv5(x6)
        outcat = torch.cat((out3, out4, out5), 1)
        out6 = self.deconv6(outcat)
        normal = self.est_normal(out6)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal


class AttExtractor(nn.Module):
    def __init__(self, batchNorm=True, c_in=3, other={}):
        super(AttExtractor, self).__init__()
        self.convA = model_utils.conv(batchNorm, 3, 128, k=3, stride=2, pad=1)
        self.convB = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.convD = model_utils.deconv(128, 64)

    def forward(self, x):
        x = x[:, 0:3, :, :]
        h_x = x.size()[2]
        w_x = x.size()[3]
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2)
        xgrad = xgrad*2
        x3 = self.convA(xgrad)
        attention_one = self.convD(x3)
        n1, c1, h1, w1 = attention_one.data.shape
        attention_one = attention_one.view(-1)
        return  attention_one, [n1,c1,h1,w1]

class AttRegressor(nn.Module):
    def __init__(self, batchNorm=True, other={}):
            super(AttRegressor, self).__init__()
            self.other = other
            self.deconv1 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)
            self.deconv4 = model_utils.deconv(128, 64)
            self.deconv5 = self._make_output(64,1, k=3, stride=1, pad=1)
            self.other = other
    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))
    def forward(self, x, shape):
        x = x.view(shape[0] , shape[1], shape[2], shape[3])
        out = self.deconv1(x)
        out = self.deconv4(out)
        attention = self.deconv5(out)
        return attention

class NormAttentionPSN(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(NormAttentionPSN, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.regressor = Regressor(batchNorm, other)
        self.attextractor = AttExtractor(batchNorm, c_in, other)
        self.attregressor = AttRegressor(batchNorm, other)
        self.c_in = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)

    def forward(self, x):
        img = x[0]
        img_split = torch.split(img, 3, 1)
        if len(x) > 1:  # Have lighting
            light = x[1]
            light_split = torch.split(light, 3, 1)

        feats6 = []
        feats10 = []
        feats13 = []
        atts = []
        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            feat6, shape6, feat10, shape10, feat13, shape13 = self.extractor(net_in)
            att, shape = self.attextractor(net_in)
            feats6.append(feat6)
            feats10.append(feat10)
            feats13.append(feat13)
            atts.append(att)

        if self.fuse_type == 'mean':
            feat_fused6 = torch.stack(feats6, 1).mean(1)
            feat_fused10 = torch.stack(feats10, 1).mean(1)
            feat_fused13 = torch.stack(feats13, 1).mean(1)
        elif self.fuse_type == 'max':
            feat_fused6, _ = torch.stack(feats6, 1).max(1)
            feat_fused10, _ = torch.stack(feats10, 1).max(1)
            feat_fused13, _ = torch.stack(feats13, 1).max(1)
            att_fused, _ = torch.stack(atts, 1).max(1)
        normal = self.regressor(feat_fused6, shape6, feat_fused10, shape10, feat_fused13, shape13)
        attentionmap = self.attregressor(att_fused, shape)
        maxx = torch.max(attentionmap).item()
        minn = torch.min(attentionmap).item()
        att_scale = maxx - minn
        [n1, c1, h1, w1] = shape
        min_shape = torch.ones_like(attentionmap)
        min_att = min_shape * minn
        attentionmap = attentionmap - min_att
        attentionmap = torch.div(attentionmap, att_scale)
        return normal, attentionmap
