import sys
import torch
import torch.nn as nn

class ResBlock(nn.Module):    
    def __init__(self,in_channel,channel):
        super(ResBlock,self).__init__()
        
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel,channel,3,padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel,in_channel,1)
        )
        
    def forward(self,x):
        out = self.conv(x)
        out += x
        return out

class ResNetCNN(nn.Module):

    def __init__(self, 
                 channel=1, 
                 channel_in=1,
#                  channel_res=32,
                 n_res_block =4, 
                 num_classes=1):
        
        super(ResNetCNN, self).__init__()
        
        self.channel_in = channel_in
        
        cnn_blocks = [
            nn.Conv2d(1, 8, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(32)
        ]
        dens_blocks = [
            nn.Linear(32, 100),
            nn.PReLU(),
#             nn.Dropout(),
            nn.Linear(100, 100),
            nn.PReLU(),
#             nn.Dropout(),
            nn.Linear(100, 100),
            nn.PReLU(),
#             nn.Dropout(),
            nn.Linear(100, 100),
            nn.PReLU(),
#             nn.Dropout(),
        ]

        
        for i in range(n_res_block):
            cnn_blocks.append(ResBlock(in_channel=32, channel=32))
        cnn_blocks.append(nn.AdaptiveAvgPool2d((1,1)))

        self.cnn_blocks = nn.Sequential(*cnn_blocks)
        self.dense_block = nn.Sequential(*dens_blocks)
        self.linear = nn.Linear(100, num_classes)
       

    def forward(self, x):
        h = self.cnn_blocks(x)
        h = h.view(h.size(0), -1)
        h = self.dense_block(h)
        out = self.linear(h)
        return out


if __name__ == '__main__':
    inputs = torch.randn(100, 1, 128, 64)
    model = ResNetCNN(num_classes=1)
    print('*'*30)
    print(f'Network structure\n')
#     summary(model, (1, 128, 64), device='cpu')
    print(model)
    print('*'*30)
    outputs = model(inputs) 
    print(f'\noutput size : {outputs.size()}\n')
#     print(torch.squeeze(outputs))   

    


    