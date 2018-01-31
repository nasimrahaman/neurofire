import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable

from .resnet import resnet101


class FCN(nn.Module):
    def __init__(self, out_channels=19, output_stride=4, mode='bilinear'):
        super(FCN, self).__init__()

        self.output_stride = output_stride
        self.ResNet = resnet101(output_stride=output_stride, pretrained=True)
        del self.ResNet.avgpool
        del self.ResNet.fc

        # FCN
        layer4_channels = 2048
        self.aspp = ASPP_Module(inplanes=layer4_channels, output_stride=output_stride)

        aspp_channels = 1280
        self.final_conv = nn.Sequential(nn.Conv2d(aspp_channels, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, out_channels, kernel_size=1, bias=True))
        initialize_weights(self.final_conv)

        self.upsample = nn.Upsample(scale_factor=output_stride, mode=mode)


    def forward(self, x):
        x = self.ResNet(x)
        x = self.aspp(x)

        # Final stage:
        x = self.final_conv(x)

        if self.output_stride > 1:
            return self.upsample(x)
        else:
            return x


class ASPP_Module(nn.Module):
    def __init__(self, inplanes, output_stride=16):
        super(ASPP_Module, self).__init__()

        scale = 16//output_stride
        planes = 256

        dilation0 = scale*6
        self.conv0 = nn.Sequential( nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=dilation0, dilation=dilation0, bias=False))
                                    # nn.BatchNorm2d(planes))
        initialize_weights(self.conv0)

        dilation1 = scale*12
        self.conv1 = nn.Sequential( nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=dilation1, dilation=dilation1, bias=False))
                                    # nn.BatchNorm2d(planes))
        initialize_weights(self.conv1)

        dilation2 = scale*18
        self.conv2 = nn.Sequential( nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=dilation2, dilation=dilation2, bias=False))
                                    # nn.BatchNorm2d(planes))
        initialize_weights(self.conv2)

        self.conv3 = nn.Sequential( nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
                                    # nn.BatchNorm2d(planes))
        initialize_weights(self.conv3)


        self.aspp_pooling = ASPP_Pooling(inplanes, planes)


    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        pool = self.aspp_pooling(x)

        return torch.cat([x3, x0, x1, x2, pool], dim=1)




class ASPP_Pooling(nn.Module):
    def __init__(self, inplanes, planes):
        super(ASPP_Pooling, self).__init__()

        self.conv = nn.Sequential(  nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
                                    # nn.BatchNorm2d(planes))
        initialize_weights(self.conv)


    def forward(self, x):
        size = [x.size(2), x.size(3)]

        pool = F.avg_pool2d(x, size)
        pool = self.conv(pool)
        return F.upsample(pool, size=size, scale_factor=None, mode='bilinear') 




def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
