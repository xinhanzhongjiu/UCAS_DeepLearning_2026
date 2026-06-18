import torch
import torch.nn as nn

def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class SegNet(nn.Module):
    def __init__(self, num_classes=12, in_channels=3):
        super().__init__()

        # ===== Encoder =====
        self.enc1 = nn.Sequential(conv_bn_relu(in_channels, 64),
                                  conv_bn_relu(64, 64))
        self.enc2 = nn.Sequential(conv_bn_relu(64, 128),
                                  conv_bn_relu(128, 128))
        self.enc3 = nn.Sequential(conv_bn_relu(128, 256),
                                  conv_bn_relu(256, 256),
                                  conv_bn_relu(256, 256))
        self.enc4 = nn.Sequential(conv_bn_relu(256, 512),
                                  conv_bn_relu(512, 512),
                                  conv_bn_relu(512, 512))
        self.enc5 = nn.Sequential(conv_bn_relu(512, 512),
                                  conv_bn_relu(512, 512),
                                  conv_bn_relu(512, 512))

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

        # ===== Decoder =====（与 encoder 对称）
        self.dec5 = nn.Sequential(conv_bn_relu(512, 512),
                                  conv_bn_relu(512, 512),
                                  conv_bn_relu(512, 512))
        self.dec4 = nn.Sequential(conv_bn_relu(512, 512),
                                  conv_bn_relu(512, 512),
                                  conv_bn_relu(512, 256))
        self.dec3 = nn.Sequential(conv_bn_relu(256, 256),
                                  conv_bn_relu(256, 256),
                                  conv_bn_relu(256, 128))
        self.dec2 = nn.Sequential(conv_bn_relu(128, 128),
                                  conv_bn_relu(128, 64))
        self.dec1 = nn.Sequential(conv_bn_relu(64, 64),
                                  nn.Conv2d(64, num_classes, 3, padding=1))

    def forward(self, x):
        # Encoder + 记录 indices 与 size
        x = self.enc1(x); s1 = x.size(); x, i1 = self.pool(x)
        x = self.enc2(x); s2 = x.size(); x, i2 = self.pool(x)
        x = self.enc3(x); s3 = x.size(); x, i3 = self.pool(x)
        x = self.enc4(x); s4 = x.size(); x, i4 = self.pool(x)
        x = self.enc5(x); s5 = x.size(); x, i5 = self.pool(x)

        # Decoder：先 unpool 再卷积
        x = self.unpool(x, i5, output_size=s5); x = self.dec5(x)
        x = self.unpool(x, i4, output_size=s4); x = self.dec4(x)
        x = self.unpool(x, i3, output_size=s3); x = self.dec3(x)
        x = self.unpool(x, i2, output_size=s2); x = self.dec2(x)
        x = self.unpool(x, i1, output_size=s1); x = self.dec1(x)
        return x