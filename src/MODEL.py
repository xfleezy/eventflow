import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
import torch
from utils import *
import torch.nn.functional as F

class pyramidNet(nn.Module):
    def __init__(self, BatchNorm=False):
        super(pyramidNet, self).__init__()
        self.batchnorm = BatchNorm
        self.module0 = basic_module(self.batchnorm)
        self.module1 = basic_module(self.batchnorm)
        self.module2 = basic_module(self.batchnorm)
        self.module3 = basic_module(self.batchnorm)
        self.predict1= nn.Sequential(nn.Conv2d(38, 2, kernel_size=3, stride=1, padding=(3 - 1) // 2),
                             nn.BatchNorm2d(2),
                             nn.Tanh())
        self.predict2=nn.Sequential(nn.Conv2d(38, 2, kernel_size=3, stride=1, padding=(3 - 1) // 2),
                             nn.BatchNorm2d(2),
                             nn.Tanh())
        self.predict3=nn.Sequential(nn.Conv2d(38, 2, kernel_size=3, stride=1, padding=(3 - 1) // 2),
                             nn.BatchNorm2d(2),
                             nn.Tanh())
        self.layer1=nn.Sequential(nn.Conv2d(38,38,kernel_size=1,stride=1,padding=0),
                                 nn.AdaptiveAvgPool2d(1),
                                 nn.Linear(1,1),
                                 nn.Linear(1,1),
                                 nn.Sigmoid())
        self.layer2=nn.Sequential(nn.Conv2d(2,2,kernel_size=1,stride=1,padding=0),
                                 nn.AdaptiveAvgPool2d(1),
                                 nn.Linear(1,1),
                                 nn.Linear(1,1),
                                 nn.Sigmoid())
        self.layer3=nn.Sequential(nn.Conv2d(2,2,kernel_size=1,stride=1,padding=0),
                                 nn.AdaptiveAvgPool2d(1),
                                 nn.Linear(1,1),
                                 nn.Linear(1,1),
                                 nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)



    def forward(self, x,gray):
       
        x0=x[0].float().cuda()
        x1=x[1].float().cuda()
        x2=x[2].float().cuda()
        gray1=gray[0].float().cuda()
        gray2=gray[1].float().cuda()
        gray3=gray[2].float().cuda()

        flow11 = torch.zeros((x0.shape[0], 2, 64, 86)).float().cuda()

        input1=torch.cat((x2,flow11),1)
        flow2=self.predict1(torch.cat((self.module1(input1),gray3),1))+flow11
       
        flow22 = F.interpolate(flow2, (flow2.shape[2] * 2, flow2.shape[3] * 2), mode='bilinear')

        flow3=self.predict2(torch.cat((self.module2(torch.cat((x1,flow22),1)),gray2),1))+flow22


        flow33=F.interpolate(flow3,(flow3.shape[2]*2,flow3.shape[3]*2),mode='bilinear')


        input=torch.cat((self.module3(torch.cat((x0,flow33),1)),gray1),1)

        flow4=self.predict3(input)+flow33

        flow4=flow4+torch.mul(flow4,self.layer2(flow4))

        if self.training:
            return flow4, flow3, flow2
        else:
            return flow4

    def weight_parameters(self):
        return [para for name, para in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [para for name, para in self.named_parameters() if 'bias' in name]

def pyramidnet(data=None):
    model = pyramidNet()
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
