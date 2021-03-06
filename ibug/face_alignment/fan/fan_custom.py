import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, use_instance_norm):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes) if use_instance_norm else nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = (nn.InstanceNorm2d(int(out_planes / 2)) if use_instance_norm
                    else nn.BatchNorm2d(int(out_planes / 2)))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = (nn.InstanceNorm2d(int(out_planes / 4)) if use_instance_norm
                    else nn.BatchNorm2d(int(out_planes / 4)))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.InstanceNorm2d(in_planes) if use_instance_norm
                                            else nn.BatchNorm2d(in_planes),
                                            nn.ReLU(True),
                                            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):
    def __init__(self, config):
        super(HourGlass, self).__init__()
        self.config = config

        self._generate_network(self.config.hg_depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.config.hg_num_features,
                                                      self.config.hg_num_features,
                                                      self.config.use_instance_norm))

        self.add_module('b2_' + str(level), ConvBlock(self.config.hg_num_features,
                                                      self.config.hg_num_features,
                                                      self.config.use_instance_norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level),ConvBlock(self.config.hg_num_features,
                                                              self.config.hg_num_features,
                                                              self.config.use_instance_norm))

        self.add_module('b3_' + str(level), ConvBlock(self.config.hg_num_features,
                                                      self.config.hg_num_features,
                                                      self.config.use_instance_norm))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        if self.config.use_avg_pool:
            low1 = F.avg_pool2d(inp, 2)
        else:
            low1 = F.max_pool2d(inp, 2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.config.hg_depth, x)


class FAN(nn.Module):
    def __init__(self, config):
        super(FAN, self).__init__()
        self.config = config

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=self.config.stem_conv_kernel_size,
                               stride=self.config.stem_conv_stride,
                               padding=self.config.stem_conv_kernel_size // 2)
        self.bn1 = nn.InstanceNorm2d(64) if self.config.use_instance_norm else nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128, self.config.use_instance_norm)
        self.conv3 = ConvBlock(128, 128, self.config.use_instance_norm)
        self.conv4 = ConvBlock(128, self.config.hg_num_features, self.config.use_instance_norm)

        # Hourglasses
        for hg_module in range(self.config.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(self.config))
            self.add_module('top_m_' + str(hg_module), ConvBlock(self.config.hg_num_features,
                                                                 self.config.hg_num_features,
                                                                 self.config.use_instance_norm))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(self.config.hg_num_features,
                                                                    self.config.hg_num_features,
                                                                    kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module),
                            nn.InstanceNorm2d(self.config.hg_num_features) if self.config.use_instance_norm
                            else nn.BatchNorm2d(self.config.hg_num_features))
            self.add_module('l' + str(hg_module), nn.Conv2d(self.config.hg_num_features,
                                                            self.config.num_landmarks,
                                                            kernel_size=1, stride=1, padding=0))

            if hg_module < self.config.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(self.config.hg_num_features,
                                                                 self.config.hg_num_features,
                                                                 kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(self.config.num_landmarks,
                                                                 self.config.hg_num_features,
                                                                 kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = self.conv2(F.relu(self.bn1(self.conv1(x)), True))
        if self.config.stem_pool_kernel_size > 1:
            if self.config.use_avg_pool:
                x = F.avg_pool2d(x, self.config.stem_pool_kernel_size)
            else:
                x = F.max_pool2d(x, self.config.stem_pool_kernel_size)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        hg_feats = []
        tmp_out = None
        for i in range(self.config.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)

            if i < self.config.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

            hg_feats.append(ll)

        # return tmp_out, x, tuple(hg_feats)
        return self._decode(tmp_out), tmp_out



    def _decode(self, heatmaps: torch.Tensor):
        heatmaps = heatmaps.contiguous()
        scores = heatmaps.max(dim=3)[0].max(dim=2)[0]

        if (self.config.radius ** 2 * heatmaps.shape[2] * heatmaps.shape[3] <
                heatmaps.shape[2] ** 2 + heatmaps.shape[3] ** 2):
            # Find peaks in all heatmaps
            m = heatmaps.view(heatmaps.shape[0] * heatmaps.shape[1], -1).argmax(1)
            # all_peaks = torch.cat(
            #     [(m / heatmaps.shape[3]).trunc().view(-1, 1), (m % heatmaps.shape[3]).view(-1, 1)], dim=1
            # ).reshape((heatmaps.shape[0], heatmaps.shape[1], 1, 1, 2)).repeat(
            #     1, 1, heatmaps.shape[2], heatmaps.shape[3], 1).float()
            all_peaks = torch.cat(
                [torch.div(m, heatmaps.shape[3], rounding_mode="trunc").view(-1, 1), (m % heatmaps.shape[3]).view(-1, 1)], dim=1
            ).reshape((heatmaps.shape[0], heatmaps.shape[1], 1, 1, 2)).repeat(
                1, 1, heatmaps.shape[2], heatmaps.shape[3], 1).float()


            # Apply masks created from the peaks
            all_indices = torch.zeros_like(all_peaks) + torch.stack(
                [
                    torch.arange(0.0, all_peaks.shape[2], device=all_peaks.device).unsqueeze(-1).repeat(1, all_peaks.shape[3]),
                    torch.arange(0.0, all_peaks.shape[3], device=all_peaks.device).unsqueeze(0).repeat(all_peaks.shape[2], 1)
                ], dim=-1)
            heatmaps = heatmaps * ((all_indices - all_peaks).norm(dim=-1) <= self.config.radius *
                                   (heatmaps.shape[2] * heatmaps.shape[3]) ** 0.5).float()

        # Prepare the indices for calculating centroids
        x_indices = (torch.zeros((*heatmaps.shape[:2], heatmaps.shape[3]), device=heatmaps.device) + torch.arange(0.5, heatmaps.shape[3], device=heatmaps.device))
        y_indices = (torch.zeros(heatmaps.shape[:3], device=heatmaps.device) + torch.arange(0.5, heatmaps.shape[2], device=heatmaps.device))

        # Finally, find centroids as landmark locations
        heatmaps = heatmaps.clamp_min(0.0)
        if self.config.gamma != 1.0:
            heatmaps = heatmaps.pow(self.config.gamma)
        m00s = heatmaps.sum(dim=(2, 3)).clamp_min(torch.finfo(heatmaps.dtype).eps)
        xs = heatmaps.sum(dim=2).mul(x_indices).sum(dim=2).div(m00s)
        ys = heatmaps.sum(dim=3).mul(y_indices).sum(dim=2).div(m00s)

        lm_info = torch.stack((xs, ys, scores), dim=-1)#.cpu().numpy()
        # return lm_info[..., :-1], lm_info[..., -1]
        return lm_info