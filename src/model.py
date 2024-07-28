import torch
import torch.backends
import torch.nn as nn
import torch.nn.functional as F

import os
import torchinfo

# Set the environment variable within the notebook
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Optionally, verify that it's set
print(os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"])
class InitLayer(nn.Module):
    def __init__(self, in_channels, out_channels)-> None:
        super(InitLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                             out_channels,
                             padding=3, 
                             kernel_size=7, 
                             stride=2,
                             bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,x)-> torch.Tensor:
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResTwoBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(ResTwoBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels= self.in_channels,
                                out_channels = self.in_channels,
                                kernel_size = 1,
                                bias=False)
        self.conv2 = nn.Conv2d(in_channels= self.in_channels,
                                out_channels= self.in_channels,
                                kernel_size= 3,
                                stride= 1,
                                bias= False,
                                padding=1)
        self.conv3 = nn.Conv2d(in_channels= self.in_channels,
                                out_channels= self.out_channels,
                                kernel_size= 1,
                                bias=False)

        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.bn2 = nn.BatchNorm2d(self.in_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        
        main = x
        shortcut = x
        
        if self.in_channels != self.out_channels:
            bs, c, h, w = shortcut.shape
            # print('start shortcut',shortcut.shape)
            extra_channels = self.out_channels - c
            # print('extra channels',extra_channels)
            channel_padding = torch.zeros((bs, extra_channels,h,w),device=x.device)
            # print('padding shape',channel_padding.shape)
            shortcut = torch.cat((shortcut,channel_padding),1)
            # print('shortcut',shortcut.shape)
    

        # print('main shape', x.shape)
        
        main = self.conv1(x)
        main = self.bn1(main)
        main = self.relu(main)
        # print('main shape', main.shape)
        
        main = self.conv2(main)
        main = self.bn2(main)
        main = self.relu(main)
        # print('main shape', main.shape)
        
        main = self.conv3(main)
        main = self.bn3(main)
        main = self.relu(main)
        # print('main shape', main.shape)
        main = main + shortcut

        return main

class DownsampleToNext(nn.Module):
    def __init__(self, in_channels, out_channels, down = False):
        super(DownsampleToNext, self).__init__()
        self.stride = 2 if down else 1
        self.c1 = nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=self.stride, bias=False, padding= 1 )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.c1(x)
        x = self.bn(x)
        return x


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.7) -> None:
        super().__init__()
        self.init1 = InitLayer(3,64)#224 ->112 -> 56 
        self.b10 = ResTwoBottleneck(64,256)
        self.change1a = DownsampleToNext(256, 64)
        self.b11 = ResTwoBottleneck(64,256)
        self.change1b = DownsampleToNext(256, 64)
        self.b12 = ResTwoBottleneck(64,256)
        
        self.downa = DownsampleToNext(256,128, down= True) #56 -> 28
        self.b20 = ResTwoBottleneck(128,512)
        self.chang2a = DownsampleToNext(512, 128)
        self.b21 = ResTwoBottleneck(128,512)
        # self.b22 = ResTwoBottleneck(128,512)
        # self.b23 = ResTwoBottleneck(128,512)
        self.downb = DownsampleToNext(512,256, down= True) #28 -> 14
        self.b30 = ResTwoBottleneck(256, 1024)
        self.downc = DownsampleToNext(1024, 512, down= True)# 14 -> 7
        self.b40 = ResTwoBottleneck(512, 1024)

        self.b50 = nn.AvgPool2d(kernel_size=7, stride=1)
        self.finalconv = nn.Conv2d(1024, num_classes, kernel_size= 1)
        self.flat = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init1(x)
        x = self.b10(x)
        x = self.change1a(x)
        x = self.b11(x)
        x = self.change1b(x)
        x = self.b12(x)

        x = self.downa(x)
        x = self.b20(x)
        x = self.chang2a(x)
        x = self.b21(x)

        x = self.downb(x)
        x = self.b30(x)
        x = self.downc(x)
        x = self.b40(x)
        x = self.b50(x)
        x = self.finalconv(x)
        x = self.flat(x)

        return x



######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)
    
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
