import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self,  hidden_dim, input_size):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size , hidden_dim)
        self.Transconv1 = nn.ConvTranspose2d(1, 1 , 3, stride=(3,2) , padding=0, output_padding=0, groups=1,bias=True, dilation=1, padding_mode='zeros')
        self.Transconv2 = nn.ConvTranspose2d(1, 1, 4, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        self.Transconv3 = nn.ConvTranspose2d(1, 1, 5, stride=1, padding=0, output_padding=0, groups=1 , bias=True, dilation=(2,1), padding_mode='zeros')
        self.bn0 = nn.BatchNorm1d(1504, affine=False)
    def forward(self, data):
        B, N, D = data.shape
        data = data.view(-1, D)
        data = self.bn0(data)
        data = data.view(B, N, D)
        lstm_out, _ = self.lstm(data)
        x=self.Transconv1(lstm_out[-1].view(1,1,12,12))
        x=self.Transconv2(x)
        tag_scores=self.Transconv3(x)

        return F.relu(tag_scores)