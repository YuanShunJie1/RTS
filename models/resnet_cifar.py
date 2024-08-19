import os
import math
import torch
import torch.nn as nn
import torchvision.models

import torch.nn.functional as F


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# you need to download the models to ~/.torch/models
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
models_dir = os.path.expanduser('~/.torch/models')
model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=12):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        feat = x.view(x.size(0), -1)
        out = self.fc(feat)
        return out, feat


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet34'])))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet50'])))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet101'])))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet152'])))
    return model






class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.momentum = 0.1
        
        self.c1=nn.Conv2d(3, 16,kernel_size=3,stride=1, padding=1)
        self.linear1=nn.Linear(4096, 10)
        self.bn1=nn.BatchNorm2d(16, momentum=self.momentum)
        
    def forward(self, x):
        h=x
        h=self.c1(h)
        h=F.relu(call_bn(self.bn1, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit=self.linear1(h)
        return logit, logit


class MMLPNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MMLPNet, self).__init__()
        self.fc1 = nn.Linear(3* 32 * 32, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 3* 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, x

class MLPNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(3* 32 * 32, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 3* 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, x
    
    
def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1, num_classes=10):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel, 64,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(128,196,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(196,16,kernel_size=3,stride=1, padding=1)
        self.linear1=nn.Linear(256, n_outputs)
        self.bn1=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn2=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn3=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn4=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn5=nn.BatchNorm2d(196, momentum=self.momentum)
        self.bn6=nn.BatchNorm2d(16, momentum=self.momentum)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.relu(call_bn(self.bn1, h))
        h=self.c2(h)
        h=F.relu(call_bn(self.bn2, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c3(h)
        h=F.relu(call_bn(self.bn3, h))
        h=self.c4(h)
        h=F.relu(call_bn(self.bn4, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c5(h)
        h=F.relu(call_bn(self.bn5, h))
        h=self.c6(h)
        h=F.relu(call_bn(self.bn6, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit=self.linear1(h)
        return logit, logit