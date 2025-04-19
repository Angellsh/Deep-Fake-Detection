import torch.nn as nn
import torchvision

class resBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(resBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.out_channels = out_channels
    def forward(self, x):
        residuals = x
        out = self.conv1(x)
        out = self.conv2(out)
        out+=residuals
        return nn.ReLU(out)
class resNet(nn.module):
    def __init__(self, block, layers, num_classes=10):
        super(resNet,self).__init__()
        self.inplanes = 64
        self.conv1=nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=3), 
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self.make_layer(block, 64, layers[0])
        self.layer1 = self.make_layer(block, 128, layers[1])
        self.layer2 = self.make_layer(block, 256, layers[2] )
        self.layer3 = self.make_layer(block, 256, layers[2] )
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc=nn.Linear(512, num_classes)



    def make_layer(self, block, planes, blocks):
        layers=[]
        layers.append(block(self.inplanes), planes)
        self.inplanes=planes
        for i in range(blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x= self.conv1(x)
        x= self.maxpool(x)
        x= self.layer0(x)
        x= self.layer1(x)
        x= self.layer2(x)
        x= self.layer3(x)
        x= self.fc(x)
        return x







        

        