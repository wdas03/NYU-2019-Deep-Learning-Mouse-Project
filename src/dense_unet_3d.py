import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import BatchNormRelu3D, BatchNormRelu3D, BatchNormReluDropout3D

class ModelDropout(nn.Module):
    def __init__(self):
        super(ModelDropout, self).__init__()
        
        self.conv1_1 = BatchNormRelu3D(1, 8, 3, 1, padding=1)
        self.conv1_2 = BatchNormRelu3D(9, 8, 3, 1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout3d(0.5)
        
        self.conv2_1 = BatchNormRelu3D(9, 16, 3, padding=1)
        self.conv2_2 = BatchNormRelu3D(25, 16, 3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout3d(0.5)
        
        self.conv3_1 = BatchNormRelu3D(25, 32, 3, padding=1)
        self.conv3_2 = BatchNormRelu3D(57, 32, 3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout3d(0.3)
        
        self.conv4_1 = BatchNormRelu3D(57, 64, 3, padding=1)
        self.conv4_2 = BatchNormRelu3D(121, 64, 3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout3d(0.3)
        
        self.conv5_1 = BatchNormRelu3D(121, 128, 3, padding=1)
        self.conv5_2 = BatchNormRelu3D(249, 128, 3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv6_1 = BatchNormRelu3D(249, 64, 3, padding=1)
        self.conv6_2 = BatchNormRelu3D(313, 64, 3, padding=1)
        
        self.conv7_1 = BatchNormRelu3D(96, 32, 3, padding=1)
        self.conv7_2 = BatchNormRelu3D(128, 32, 3, padding=1)
        
        self.conv8_1 = BatchNormRelu3D(48, 16, 3, padding=1)
        self.conv8_2 = BatchNormRelu3D(64, 16, 3, padding=1)
        
        self.conv9_1 = BatchNormRelu3D(24, 8, 3, padding=1)
        self.conv9_2 = BatchNormReluDropout3D(32, 8, 3, padding=1)
        
        self.conv10 = nn.Conv3d(32, 3, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        
        # Transpose convolutions
        self.up1 = nn.ConvTranspose3d(249, 128, 2, 2)
        self.up2 = nn.ConvTranspose3d(313, 64, 2, 2)
        self.up3 = nn.ConvTranspose3d(128, 32, 2, 2)
        self.up4 = nn.ConvTranspose3d(64, 16, 2, 2)
        
    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conc1_1 = torch.cat([x, conv1_1], dim=1)
        conv1_2 = self.conv1_2(conc1_1)
        conc1_2 = torch.cat([x, conv1_2], dim=1)
        pool1 = self.drop1(self.pool1(conc1_2))
        
        conv2_1 = self.conv2_1(pool1)
        conc2_1 = torch.cat([conv2_1, pool1], dim=1)
        conv2_2 = self.conv2_2(conc2_1)
        conc2_2 = torch.cat([pool1, conv2_2], dim=1)
        pool2 = self.drop2(self.pool2(conc2_2))
        
        conv3_1 = self.conv3_1(pool2)
        conc3_1 = torch.cat([conv3_1, pool2], dim=1)
        conv3_2 = self.conv3_2(conc3_1)
        conc3_2 = torch.cat([pool2, conv3_2], dim=1)
        pool3 = self.drop3(self.pool3(conc3_2))
        
        conv4_1 = self.conv4_1(pool3)
        conc4_1 = torch.cat([conv4_1, pool3], dim=1)
        conv4_2 = self.conv4_2(conc4_1)
        conc4_2 = torch.cat([pool3, conv4_2], dim=1)
        pool4 = self.drop4(self.pool4(conc4_2))
        
        conv5_1 = self.conv5_1(pool4)
        conc5_1 = torch.cat([pool4, conv5_1], dim=1)
        conv5_2 = self.conv5_2(conc5_1)
        conc5_2 = torch.cat([pool4, conv5_2], dim=1)
        
        up1 = self.up1(conc5_2)
        up6 = torch.cat([up1, conc4_2], dim=1)
        conv6_1 = self.conv6_1(up6)
        conc6_1 = torch.cat([conv6_1, up6], dim=1)
        conv6_2 = self.conv6_2(conc6_1)
        conc6_2 = torch.cat([up6, conv6_2], dim=1)
        
        up2 = self.up2(conc6_2)
        up7 = torch.cat([up2, conv3_2], dim=1)
        conv7_1 = self.conv7_1(up7)
        conc7_1 = torch.cat([conv7_1, up7], dim=1)
        conv7_2 = self.conv7_2(conc7_1)
        conc7_2 = torch.cat([up7, conv7_2], dim=1)
        
        up3 = self.up3(conc7_2)
        up8 = torch.cat([up3, conv2_2], dim=1)
        conv8_1 = self.conv8_1(up8)
        conc8_1 = torch.cat([conv8_1, up8], dim=1)
        conv8_2 = self.conv8_2(conc8_1)
        conc8_2 = torch.cat([up8, conv8_2], dim=1)
        
        up4 = self.up4(conc8_2)
        up9 = torch.cat([up4, conv1_2], dim=1)
        conv9_1 = self.conv9_1(up9)
        conc9_1 = torch.cat([conv9_1, up9], dim=1)
        conv9_2 = self.conv9_2(conc9_1)
        conc9_2 = torch.cat([up9, conv9_2], dim=1)
        
        conv10 = self.conv10(conc9_2)
        out = self.softmax(conv10)
        
        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1_1 = BatchNormRelu3D(1, 8, 3, 1, padding=1)
        self.conv1_2 = BatchNormRelu3D(9, 8, 3, 1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv2_1 = BatchNormRelu3D(9, 16, 3, padding=1)
        self.conv2_2 = BatchNormRelu3D(25, 16, 3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv3_1 = BatchNormRelu3D(25, 32, 3, padding=1)
        self.conv3_2 = BatchNormRelu3D(57, 32, 3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv4_1 = BatchNormRelu3D(57, 64, 3, padding=1)
        self.conv4_2 = BatchNormRelu3D(121, 64, 3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv5_1 = BatchNormRelu3D(121, 128, 3, padding=1)
        self.conv5_2 = BatchNormRelu3D(249, 128, 3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv6_1 = BatchNormRelu3D(249, 64, 3, padding=1)
        self.conv6_2 = BatchNormRelu3D(313, 64, 3, padding=1)
        
        self.conv7_1 = BatchNormRelu3D(96, 32, 3, padding=1)
        self.conv7_2 = BatchNormRelu3D(128, 32, 3, padding=1)
        
        self.conv8_1 = BatchNormRelu3D(48, 16, 3, padding=1)
        self.conv8_2 = BatchNormRelu3D(64, 16, 3, padding=1)
        
        self.conv9_1 = BatchNormRelu3D(24, 8, 3, padding=1)
        self.conv9_2 = BatchNormRelu3D(32, 8, 3, padding=1)
        
        self.conv10 = nn.Conv3d(32, 3, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        
        # Transpose convolutions
        self.up1 = nn.ConvTranspose3d(249, 128, 2, 2)
        self.up2 = nn.ConvTranspose3d(313, 64, 2, 2)
        self.up3 = nn.ConvTranspose3d(128, 32, 2, 2)
        self.up4 = nn.ConvTranspose3d(64, 16, 2, 2)
        
    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conc1_1 = torch.cat([x, conv1_1], dim=1)
        conv1_2 = self.conv1_2(conc1_1)
        conc1_2 = torch.cat([x, conv1_2], dim=1)
        pool1 = self.pool1(conc1_2)
        
        conv2_1 = self.conv2_1(pool1)
        conc2_1 = torch.cat([conv2_1, pool1], dim=1)
        conv2_2 = self.conv2_2(conc2_1)
        conc2_2 = torch.cat([pool1, conv2_2], dim=1)
        pool2 = self.pool2(conc2_2)
        
        conv3_1 = self.conv3_1(pool2)
        conc3_1 = torch.cat([conv3_1, pool2], dim=1)
        conv3_2 = self.conv3_2(conc3_1)
        conc3_2 = torch.cat([pool2, conv3_2], dim=1)
        pool3 = self.pool3(conc3_2)
        
        conv4_1 = self.conv4_1(pool3)
        conc4_1 = torch.cat([conv4_1, pool3], dim=1)
        conv4_2 = self.conv4_2(conc4_1)
        conc4_2 = torch.cat([pool3, conv4_2], dim=1)
        pool4 = self.pool4(conc4_2)
        
        conv5_1 = self.conv5_1(pool4)
        conc5_1 = torch.cat([pool4, conv5_1], dim=1)
        conv5_2 = self.conv5_2(conc5_1)
        conc5_2 = torch.cat([pool4, conv5_2], dim=1)
        
        up1 = self.up1(conc5_2)
        up6 = torch.cat([up1, conc4_2], dim=1)
        conv6_1 = self.conv6_1(up6)
        conc6_1 = torch.cat([conv6_1, up6], dim=1)
        conv6_2 = self.conv6_2(conc6_1)
        conc6_2 = torch.cat([up6, conv6_2], dim=1)
        
        up2 = self.up2(conc6_2)
        up7 = torch.cat([up2, conv3_2], dim=1)
        conv7_1 = self.conv7_1(up7)
        conc7_1 = torch.cat([conv7_1, up7], dim=1)
        conv7_2 = self.conv7_2(conc7_1)
        conc7_2 = torch.cat([up7, conv7_2], dim=1)
        
        up3 = self.up3(conc7_2)
        up8 = torch.cat([up3, conv2_2], dim=1)
        conv8_1 = self.conv8_1(up8)
        conc8_1 = torch.cat([conv8_1, up8], dim=1)
        conv8_2 = self.conv8_2(conc8_1)
        conc8_2 = torch.cat([up8, conv8_2], dim=1)
        
        up4 = self.up4(conc8_2)
        up9 = torch.cat([up4, conv1_2], dim=1)
        conv9_1 = self.conv9_1(up9)
        conc9_1 = torch.cat([conv9_1, up9], dim=1)
        conv9_2 = self.conv9_2(conc9_1)
        conc9_2 = torch.cat([up9, conv9_2], dim=1)
        
        conv10 = self.conv10(conc9_2)
        out = self.softmax(conv10)
        
        return out