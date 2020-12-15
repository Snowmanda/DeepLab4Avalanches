import torch
import torch.nn as nn
from torch.nn.functional import grid_sample
from torch.utils import model_zoo
from segmentation_models_pytorch.encoders.resnet import resnet_encoders
from modeling.reusable_blocks import Bottleneck, DeformableBlock, SeBlock
from torchvision.models.resnet import BasicBlock
from kornia.filters.sobel import SpatialGradient
from torchvision.models.resnet import ResNet, conv1x1
from segmentation_models_pytorch.encoders import _utils as utils


class AvanetBackbone(nn.Module):

    def __init__(self, groups=1, width_per_group=64, norm_layer=None, replace_stride_with_dilation=False,
                 no_blocks=(3, 3, 3, 2), deformable=True):
        super(AvanetBackbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 62, kernel_size=7, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        ch = [3, 64, 128, 256, 512, 512]
        self.out_channels = ch

        self.layer1 = self._make_layer(ch[1], ch[2], no_blocks[0], stride=2)
        self.layer2 = self._make_layer(ch[2], ch[3], no_blocks[1], stride=2, deformable=deformable)
        self.layer3 = self._make_layer(ch[3], ch[4], no_blocks[2], stride=2, deformable=deformable)
        if replace_stride_with_dilation:
            self.layer4 = self._make_layer(ch[4], ch[5], no_blocks[3], dilation=2, deformable=deformable)
        else:
            self.layer4 = self._make_layer(ch[4], ch[5], no_blocks[3], stride=2, deformable=deformable)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, inplanes, planes, blocks, stride=1, dilation=1, deformable=False):
        norm_layer = self._norm_layer
        block = Bottleneck if not deformable else DeformableBlock

        layers = []
        layers.append(block(inplanes, planes, stride, self.groups,
                            self.base_width, dilation, norm_layer))
        for _ in range(1, blocks):
            layers.append(SeBlock(planes, planes, groups=self.groups,
                                  base_width=self.base_width, dilation=1,
                                  norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, grads):
        features = [nn.Identity()]

        x = self.conv1(x)
        x = torch.cat([x, grads], dim=1)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return features


class AdaptedResnet(ResNet):
    def __init__(self, depth=5, pretrained=True):
        super().__init__(block=BasicBlock, layers=[3, 4, 6, 3])
        if pretrained:
            settings = resnet_encoders['resnet34']["pretrained_settings"]['imagenet']
            self.load_state_dict(model_zoo.load_url(settings["url"]))

        self._depth = depth
        self.out_channels = (3, 64, 64, 128, 256, 512)
        self._in_channels = 3

        del self.fc
        del self.avgpool

        self.conv1.stride = (1, 1)
        self.layer1[0].conv1.stride = (2, 2)
        self.layer1[0].downsample = nn.Sequential(
                conv1x1(self.layer1[0].conv1.in_channels, self.layer1[0].conv1.out_channels * self.layer1[0].expansion, 2),
                nn.BatchNorm2d(self.layer1[0].conv1.out_channels * self.layer1[0].expansion),
            )

        self.stages = [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

        self.make_dilated(
            stage_list=[5],
            dilation_list=[2],
        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features

    def make_dilated(self, stage_list, dilation_list):
        stages = self.stages
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            utils.replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )
