import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseVNet(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
        super(DenseVNet, self).__init__()
        self.downsample_layer = nn.Sequential(
            nn.AvgPool3d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
        
        self.initial_features = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=5, padding=2, dilation=1, stride=2),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )
        
        self.high_res_block = DenseFeatureStack(25, dense_out_channels=4, num_layers=5)
        
        self.down2 = nn.Sequential(
            nn.Conv3d(45, 24, kernel_size=3, dilation=1, stride=2),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )
        
        self.medium_res_block = DenseFeatureStack(24, dense_out_channels=8, num_layers=10)
        self.down4 = nn.Sequential(
            nn.Conv3d(104, 24, kernel_size=3, dilation=1, stride=2),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )
        
        self.low_res_block = DenseFeatureStack(24, dense_out_channels=16, num_layers=10)
        
        self.skip_high_res = nn.Sequential(
            nn.Conv3d(45, 12, kernel_size=3, stride=1, dilation=1),
            nn.BatchNorm3d(12),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.5)
        )
        self.skip_medium_res = nn.Sequential(
            nn.Conv3d(104, 24, kernel_size=3, stride=1, dilation=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.3)
        )
        self.skip_low_res = nn.Sequential(
            nn.Conv3d(184, 24, kernel_size=3, stride=1, dilation=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2)
        )
        
        self.classification = nn.Conv3d(60, self.num_classes, kernel_size=3, stride=1, dilation=1, padding=1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        sub2 = self.downsample_layer(x)
        initial_conv = self.initial_features(x)
        
        inp = torch.cat([sub2, initial_conv], dim=1)
        #inp = initial_conv
        
        high_res_block = self.high_res_block(inp)
        high_res_down = self.down2(high_res_block)
        
        medium_res_block = self.medium_res_block(high_res_down)
        medium_res_down = self.down4(medium_res_block)
        
        low_res_block = self.low_res_block(medium_res_down)
        
        skip_high_res = self.skip_high_res(high_res_block)
        up_shape = skip_high_res.shape[2:]
        
        skip_medium_res = self.skip_medium_res(medium_res_block)
        skip_medium_res_up = F.interpolate(skip_medium_res, size=up_shape, mode='trilinear', align_corners=True)
        
        skip_low_res = self.skip_low_res(low_res_block)
        skip_low_res_up = F.interpolate(skip_low_res, size=up_shape, mode='trilinear', align_corners=True)
        
        concat_features = torch.cat([skip_high_res, skip_medium_res_up, skip_low_res_up], dim=1)
        
        concat_features = F.interpolate(concat_features, size=(256, 256, 256), mode='trilinear', align_corners=True)
        classes = self.classification(concat_features)
        #print(classes)
        out = self.softmax(classes)
        #print(out.shape)
        
        return out
        
class DenseFeatureStack(nn.Module):
    def __init__(self, initial_in_channels, dense_out_channels, num_layers):
        super(DenseFeatureStack, self).__init__()
        
        self.layers = nn.ModuleList()
        current_in_channels = initial_in_channels
        for i in range(num_layers):
            # determine number of dilations
            dilation = None
            if i == 2:
                dilation = 3
            elif i == 3:
                dilation = 9
            else:
                dilation = 1
            # define convolution to add
            conv = None
            if i == 0:
                conv = nn.Sequential(
                    nn.Conv3d(initial_in_channels, dense_out_channels, kernel_size=3, stride=1, dilation=dilation, padding=1),
                    nn.BatchNorm3d(dense_out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                conv = nn.Sequential(
                    nn.Conv3d(initial_in_channels + i * dense_out_channels, dense_out_channels, kernel_size=3, stride=1, dilation=dilation, padding=dilation),
                    nn.BatchNorm3d(dense_out_channels),
                    nn.ReLU(inplace=True)
                )
            
            self.layers.append(conv)
    
    def forward(self, x):
        outputs = [x]
        #current_output = x
        for i, conv in enumerate(self.layers):
            # perform convolution on data
            #current_output = outputs[-1]
            conv_out = None
            if i == 0:
                conv_out = conv(outputs[-1])
            else:
                conv_out = conv(torch.cat(outputs, dim=1))
            #channel_wise_cat = torch.cat([conv_out, current_output], dim=1)
            outputs.append(conv_out)
            #current_output
        #for i in outputs:
        #    print(i.shape)
        
        return torch.cat(outputs, dim=1)
        