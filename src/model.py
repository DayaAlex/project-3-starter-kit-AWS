import torch
import torch.nn as nn
import torch.nn.functional as F

class InitLayer(nn.Module):
    def __init__(self, in_channels, out_channels=13)-> None:
        super(InitLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,padding=1, kernel_size=3, stride=2)
        self.bn = nn.BatchNorm2d(in_channels + out_channels)
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,x)-> torch.tensor:
        skip = x
        skip = self.max(x)

        main = x
        main = self.conv(main)
        
        return F.relu(self.bn(torch.cat((main, skip), dim=1)))

#      from mobilenet-v2(inverted residual and linear bottleneck)
#       
# stride = 2(downsampling or channelchanges)               stride = 1(skip connections present)
#       input                                                                  input
#         |                                                                     /\
#         |                                                                    /  \
#      1x1, BN, Relu6                                                         |  1x1, BN, Relu6  
#      3x3, dwise, BN, Relu6                                                  | 3x3, dwise, BN, Relu6
#    1x1, BN, linear activation                                               | 1x1, BN, linear activation  
#         |                                                                   \   |
#         |                                                                   \  /
#         |                                                                    +
#       output                                                               output
class Bottleneck(nn.Module):
    def __init__(self,in_channels, out_channels, expansion_factor=6, downsample = False) -> None:
        super(Bottleneck,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_factor = expansion_factor
        self.downsample = downsample
        self.neck = self.out_channels*self.expansion_factor

        self.bn1 = nn.BatchNorm2d(self.neck)
        self.bn2 = nn.BatchNorm2d(self.neck)
        self.bn3 = nn.BatchNorm2d(self.out_channels)

        self.stride = 2 if self.downsample else 1

        # 1x1 conv for expansion, nonlinear activation
        self.conv1 = nn.Conv2d(
            in_channels= self.in_channels,
            out_channels= self.neck, 
            kernel_size= 1,
            stride=1)
        # depthwise convolution, nonlinear activation
        self.dconv = nn.Conv2d(
            in_channels= self.neck, 
            out_channels= self.neck,
            padding=1,
            kernel_size= 3,
            stride= self.stride,
            groups=self.neck,
            )
        # 1 x1 conv for channel mixing, linear activation
        self.conv2 = nn.Conv2d(
            in_channels= self.neck,
            out_channels= self.out_channels,
            kernel_size= 1,
            stride= 1)
        
        # if self.downsample or self.in_channels!= self.out_channels:
        #     # self.shortcut = nn.Sequential(
        #     #     # nn.MaxPool2d(kernel_size= 2, 
        #     #     #           stride=2),
        #     #     nn.Conv2d(in_channels=self. in_channels,
        #     #               out_channels=self.out_channels,
        #     #               kernel_size= 2, 
        #     #               stride=2),
        #     #     nn.BatchNorm2d(self.out_channels),
        #     # )
        #     self.shortcut = None
        # else:
        #     self.shortcut = nn.Identity()
    
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        
        
        #No need to pad zeros to tensor as 1x1 conv is done. procedure for maxpool2d shortcuts
        #_, _, h, w = x.shape()
        #  if self.downsample:
        #     shortcut = self.shortcut(x)
        #     extra_channels = self.out_channels - self.in_channels
        #     channel_padding = torch.zeros((extra_channels,h,w))
        #     shortcut = shortcut.to(device)
        #     shortcut = torch.cat((shortcut,channel_padding),1)
        # else:
        #     shortcut = x

        main = x
        
        main = F.relu6(self.bn1(self.conv1(main)))
        main = F.relu6(self.bn2(self.dconv(main)))
        main = self.conv2(main)

        if self.downsample or self.in_channels != self.out_channels:
            main = self.bn3(main)
        else:
            shortcut = x
            main = self.bn3(main + shortcut)
        return main

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.7) -> None:
        super().__init__()
        self.init1 = InitLayer(3,13)#224 -> 112 
        self.b10 = Bottleneck(16,64, downsample=True)#112 -> 112
        self.b11 = Bottleneck(64,64)# 112 -> 112
        self.b12 = Bottleneck(64,64)
        self.b13 = Bottleneck(64,64)
        self.b14 = Bottleneck(64,64)
        self.b15 = Bottleneck(64,64)
        self.b16 = Bottleneck(64,64)
        self.b20 = Bottleneck(64,128)# 112 -> 56
        self.b21 = Bottleneck(128,128)
        self.b22 = Bottleneck(128,128)
        self.b23 = Bottleneck(128,128)
        self.b24 = Bottleneck(128,128)
        self.b25 = Bottleneck(128,128)
        self.b30 = nn.AvgPool2d(kernel_size=56, stride=1)# 56 -> 1
        self.finalconv = nn.Conv2d(128, num_classes, kernel_size= 1)
        self.flat = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        x = self.init1(x)
        print(x.shape)
        x = self.b10(x)
        print(x.shape)
        x = self.b11(x)
        print(x.shape)
        x = self.b12(x)
        print(x.shape)
        x = self.b13(x)
        print(x.shape)
        x = self.b14(x)
        print(x.shape)
        x = self.b15(x)
        print(x.shape)
        x = self.b16(x)
        print(x.shape)
        x = self.b20(x)
        print(x.shape)
        x = self.b21(x)
        print(x.shape)
        x = self.b22(x)
        print(x.shape)
        x = self.b23(x)
        print(x.shape)
        x = self.b24(x)
        print(x.shape)
        x = self.b25(x)
        print(x.shape)
        x = self.b30(x)
        print(x.shape)
        x = self.finalconv(x)
        print(x.shape)
        x = self.flat(x)
        print(x.shape)
        print('hello')
        x = self.softmax(x)
        print(x.shape)
        print('hello2')
        
        
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
