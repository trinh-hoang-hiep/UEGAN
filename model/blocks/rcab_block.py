import torch.nn as nn 


class CALayer(nn.Module): 
    def __init__(self, channel, reduction=16): 
        super(CALayer, self).__init__() 
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        
        self.conv_du = nn.Sequential( 
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), 
                nn.Sigmoid() 
        )

    def forward(self, x): 
        y = self.avg_pool(x) 
        y = self.conv_du(y) 
        return x * y 

class RCAB(nn.Module):  
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1): 

        super(RCAB, self).__init__() 
        modules_body = [] 
        for i in range(2): 
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias)) 
            if bn:  
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:  
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction)) 
        self.body = nn.Sequential(*modules_body) 
        self.res_scale = res_scale 

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True): 
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias) 

    def forward(self, x): 
        res = self.body(x) 
        res += x 
        return res 
