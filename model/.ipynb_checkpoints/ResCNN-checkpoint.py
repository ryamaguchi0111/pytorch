import sys
import torch
import torch.nn as nn
from torchsummary import summary

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
                 channel_in=1, 
                 channel=32,
                 channel_res=32,
                 n_res_block=4,
                 n_cnn_block=2,
                 n_dense_block=3,
                 n_dense_hidden=100,
                 num_classes=1,
                 is_bn_dense=False
                ):
        
        super(ResNetCNN, self).__init__()
        
        self.channel_in = channel_in
        
        a = [1]
        a.extend(2**i for i in range(n_cnn_block-1))
        div_factor = sorted(a, reverse=True)
        cnn_blocks = []
        for i in range(n_cnn_block):
            channel_out = channel//div_factor[i]
            if i!=n_cnn_block-1:
                cnn_blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
                cnn_blocks.append(nn.BatchNorm2d(channel_out))
                cnn_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2,padding=0))
                cnn_blocks.append(nn.ReLU(inplace=True))
                channel_in=channel_out
            else:
                cnn_blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
                cnn_blocks.append(nn.BatchNorm2d(channel_out))
            
            
#         cnn_blocks = [ # CNNは小さくても良い（１つ）
#             nn.Conv2d(1, 8, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
#             nn.BatchNorm2d(8),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(8, 16, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
#             nn.BatchNorm2d(16),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
#             nn.BatchNorm2d(32)
#         ]
        dens_blocks = [nn.Linear(channel, n_dense_hidden)]
        for i in range(n_dense_block):
            if i!=0:
                dens_blocks.append(nn.Linear(n_dense_hidden,n_dense_hidden))
            if is_bn_dense:
                dens_blocks.append(nn.BatchNorm1d(n_dense_hidden))
            dens_blocks.append(nn.PReLU())
        
        
        for i in range(n_res_block):
            cnn_blocks.append(ResBlock(in_channel=channel, channel=channel_res))
        cnn_blocks.append(nn.AdaptiveAvgPool2d((1,1)))

        self.cnn_blocks = nn.Sequential(*cnn_blocks)
        self.dense_block = nn.Sequential(*dens_blocks)
        self.linear = nn.Linear(n_dense_hidden, num_classes)
       

    def forward(self, x):
        h = self.cnn_blocks(x)
        h = h.view(h.size(0), -1)
        h = self.dense_block(h)
        out = self.linear(h)
        return out
    


if __name__ == '__main__':
    inputs = torch.randn(100, 1, 128, 64)
    model = ResNetCNN(channel_in=1, 
                      channel=32,
                      channel_res=32,
                      n_res_block=4,
                      n_cnn_block=3,
                      n_dense_hidden=100,
                      num_classes=1,
                      is_bn_dense=True)
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)
            torch.nn.init.uniform_(m.bias, a=-1.0, b=0.0)
    model.apply(init_weights)
    print('*'*30)
    print(f'Network structure\n')
    summary(model, (1, 128, 64), device='cpu')
    print(model)
    print('*'*30)
    outputs = model(inputs) 
    print(f'\noutput size : {outputs.size()}\n')
#     print(torch.squeeze(outputs))   

    


    