import torch
import torch.nn as nn
import torch.nn.functional as F

# Dense VNet Architecture used as the refinement network to segment BV
import torch
import torch.nn as nn
import torch.nn.functional as F

class RefinementModel_ELU(nn.Module):
    def __init__(self, num_classes, final_shape=(128, 128, 128)):
        super(RefinementModel_ELU, self).__init__()
        
        self.num_classes = num_classes
        self.final_shape = final_shape
        
        self.downsample_layer = nn.Sequential(
            nn.AvgPool3d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace=True)
        )
        
        self.initial_features = nn.Sequential(
            nn.Conv3d(2, 24, kernel_size=5, padding=2, dilation=1, stride=2),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )
        
        self.high_res_block = DenseFeatureStack(26, dense_out_channels=4, num_layers=5)
        
        self.down2 = nn.Sequential(
            nn.Conv3d(46, 24, kernel_size=3, dilation=1, stride=2),
            nn.BatchNorm3d(24),
            nn.ELU(inplace=True)
        )
        
        self.medium_res_block = DenseFeatureStack(24, dense_out_channels=8, num_layers=10)
        self.down4 = nn.Sequential(
            nn.Conv3d(104, 24, kernel_size=3, dilation=1, stride=2),
            nn.BatchNorm3d(24),
            nn.ELU(inplace=True)
        )
        
        self.low_res_block = DenseFeatureStack(24, dense_out_channels=16, num_layers=10)
        
        self.skip_high_res = nn.Sequential(
            nn.Conv3d(46, 12, kernel_size=3, stride=1, dilation=1),
            nn.BatchNorm3d(12),
            nn.ELU(inplace=True),
            nn.Dropout3d(0.5)
        )
        self.skip_medium_res = nn.Sequential(
            nn.Conv3d(104, 24, kernel_size=3, stride=1, dilation=1),
            nn.BatchNorm3d(24),
            nn.ELU(inplace=True),
            nn.Dropout3d(0.3)
        )
        self.skip_low_res = nn.Sequential(
            nn.Conv3d(184, 24, kernel_size=3, stride=1, dilation=1),
            nn.BatchNorm3d(24),
            nn.ELU(inplace=True),
            nn.Dropout3d(0.2)
        )
        
        self.classification = nn.Conv3d(60, self.num_classes, kernel_size=3, stride=1, dilation=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        sub2 = self.downsample_layer(x)
        initial_conv = self.initial_features(x)
        
        inp = torch.cat([sub2, initial_conv], dim=1)
        #inp = initial_conv
        
        high_res_block = self.high_res_block(inp)
        del inp
        high_res_down = self.down2(high_res_block)
        
        medium_res_block = self.medium_res_block(high_res_down)
        del high_res_down
        medium_res_down = self.down4(medium_res_block)
        
        low_res_block = self.low_res_block(medium_res_down)
        del medium_res_down
        
        skip_high_res = self.skip_high_res(high_res_block)
        up_shape = skip_high_res.shape[2:]
        
        skip_medium_res = self.skip_medium_res(medium_res_block)
        skip_medium_res_up = F.interpolate(skip_medium_res, size=up_shape, mode='trilinear', align_corners=True)
        del skip_medium_res
        
        skip_low_res = self.skip_low_res(low_res_block)
        skip_low_res_up = F.interpolate(skip_low_res, size=up_shape, mode='trilinear', align_corners=True)
        del skip_low_res
        
        concat_features = torch.cat([skip_high_res, skip_medium_res_up, skip_low_res_up], dim=1)
        
        concat_features = F.interpolate(concat_features, size=self.final_shape, mode='trilinear', align_corners=True)
        classes = self.classification(concat_features)
        #print(classes)
        out = self.sigmoid(classes)
        #print(out.shape)
        
        return out

class RefinementModel(nn.Module):
    def __init__(self, num_classes, final_shape=(128, 128, 128)):
        super(RefinementModel, self).__init__()
        
        self.num_classes = num_classes
        self.final_shape = final_shape
        
        self.downsample_layer = nn.Sequential(
            nn.AvgPool3d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace=True)
        )
        
        self.initial_features = nn.Sequential(
            nn.Conv3d(2, 24, kernel_size=5, padding=2, dilation=1, stride=2),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )
        
        self.high_res_block = DenseFeatureStack(26, dense_out_channels=4, num_layers=5)
        
        self.down2 = nn.Sequential(
            nn.Conv3d(46, 24, kernel_size=3, dilation=1, stride=2),
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
            nn.Conv3d(46, 12, kernel_size=3, stride=1, dilation=1),
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
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        sub2 = self.downsample_layer(x)
        initial_conv = self.initial_features(x)
        
        inp = torch.cat([sub2, initial_conv], dim=1)
        #inp = initial_conv
        
        high_res_block = self.high_res_block(inp)
        del inp
        high_res_down = self.down2(high_res_block)
        
        medium_res_block = self.medium_res_block(high_res_down)
        del high_res_down
        medium_res_down = self.down4(medium_res_block)
        
        low_res_block = self.low_res_block(medium_res_down)
        del medium_res_down
        
        skip_high_res = self.skip_high_res(high_res_block)
        up_shape = skip_high_res.shape[2:]
        
        skip_medium_res = self.skip_medium_res(medium_res_block)
        skip_medium_res_up = F.interpolate(skip_medium_res, size=up_shape, mode='trilinear', align_corners=True)
        del skip_medium_res
        
        skip_low_res = self.skip_low_res(low_res_block)
        skip_low_res_up = F.interpolate(skip_low_res, size=up_shape, mode='trilinear', align_corners=True)
        del skip_low_res
        
        concat_features = torch.cat([skip_high_res, skip_medium_res_up, skip_low_res_up], dim=1)
        
        concat_features = F.interpolate(concat_features, size=self.final_shape, mode='trilinear', align_corners=True)
        classes = self.classification(concat_features)
        #print(classes)
        out = self.sigmoid(classes)
        #print(out.shape)
        
        return out
    
class RefinementModel_NoDown(nn.Module):
    def __init__(self, num_classes, final_shape=(128, 128, 128)):
        super(RefinementModel_NoDown, self).__init__()
        
        self.num_classes = num_classes
        self.final_shape = final_shape
        
        self.initial_features = nn.Sequential(
            nn.Conv3d(2, 24, kernel_size=5, padding=2, dilation=1, stride=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )
        
        self.high_res_block = DenseFeatureStack(24, dense_out_channels=4, num_layers=5)
        
        self.down2 = nn.Sequential(
            nn.Conv3d(44, 24, kernel_size=3, dilation=1, stride=2),
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
            nn.Conv3d(44, 12, kernel_size=3, stride=1, dilation=1),
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
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        initial_conv = self.initial_features(x)
        #inp = initial_conv
        
        high_res_block = self.high_res_block(initial_conv)
        high_res_down = self.down2(high_res_block)
        
        medium_res_block = self.medium_res_block(high_res_down)
        del high_res_down
        medium_res_down = self.down4(medium_res_block)
        
        low_res_block = self.low_res_block(medium_res_down)
        del medium_res_down
        
        skip_high_res = self.skip_high_res(high_res_block)
        up_shape = skip_high_res.shape[2:]
        
        skip_medium_res = self.skip_medium_res(medium_res_block)
        skip_medium_res_up = F.interpolate(skip_medium_res, size=up_shape, mode='trilinear', align_corners=True)
        del skip_medium_res
        
        skip_low_res = self.skip_low_res(low_res_block)
        skip_low_res_up = F.interpolate(skip_low_res, size=up_shape, mode='trilinear', align_corners=True)
        del skip_low_res
        
        concat_features = torch.cat([skip_high_res, skip_medium_res_up, skip_low_res_up], dim=1)
        
        concat_features = F.interpolate(concat_features, size=self.final_shape, mode='trilinear', align_corners=True)
        classes = self.classification(concat_features)
        #print(classes)
        out = self.sigmoid(classes)
        #print(out.shape)
        
        return out
    
class DenseFeatureStack_ELU(nn.Module):
    def __init__(self, initial_in_channels, dense_out_channels, num_layers):
        super(DenseFeatureStack_ELU, self).__init__()
        
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
                    nn.ELU(inplace=True)
                )
            else:
                conv = nn.Sequential(
                    nn.Conv3d(initial_in_channels + i * dense_out_channels, dense_out_channels, kernel_size=3, stride=1, dilation=dilation, padding=dilation),
                    nn.BatchNorm3d(dense_out_channels),
                    nn.ELU(inplace=True)
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
        
        