import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpBlock, self).__init__()
        self.scale_factor = scale_factor

        # For scale_factor=4, use kernel_size=8, padding=2
        kernel_size = 8
        padding = 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=scale_factor, padding=padding)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        h0 = self.conv(x)
        h0 = self.prelu(h0)

        l0 = self.up(h0)
        l0 = self.prelu(l0)

        # Ensure l0 has the same size as x
        if l0.size() != x.size():
            l0 = F.interpolate(l0, size=x.size()[2:], mode='bilinear', align_corners=False)

        e = l0 - x
        h1 = self.conv(e)

        return h0 + h1


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(DownBlock, self).__init__()
        self.scale_factor = scale_factor

        # For scale_factor=4, use kernel_size=8, padding=2
        kernel_size = 8
        padding = 2

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Conv2d(out_channels, in_channels, kernel_size,
                             stride=scale_factor, padding=padding)
        self.prelu = nn.PReLU()

    def forward(self, x):
        l0 = self.up(x)
        l0 = self.prelu(l0)

        h0 = self.conv(l0)
        h0 = self.prelu(h0)

        # Ensure h0 has the same size as x
        if h0.size() != x.size():
            h0 = F.interpolate(h0, size=x.size()[2:], mode='bilinear', align_corners=False)

        e = h0 - x
        l1 = self.up(e)

        return l0 + l1


class DBPN(nn.Module):
    def __init__(self, num_channels=3, base_channels=64, feat_channels=256, scale_factor=4, num_stages=7):
        super(DBPN, self).__init__()
        self.scale_factor = scale_factor

        # Initial feature extraction with more channels
        self.initial = nn.Sequential(
            nn.Conv2d(num_channels, base_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1),
            nn.PReLU()
        )

        # High-frequency feature extraction
        self.high_freq = nn.Sequential(
            nn.Conv2d(num_channels, base_channels, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        # Back-projection stages
        self.up_blocks = nn.ModuleList([
            UpBlock(base_channels, feat_channels, scale_factor) for _ in range(num_stages)
        ])
        self.down_blocks = nn.ModuleList([
            DownBlock(feat_channels, base_channels, scale_factor) for _ in range(num_stages - 1)
        ])

        # Feature processing before upsampling
        self.pre_upsample = nn.Sequential(
            nn.Conv2d(feat_channels * num_stages, 256, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # Final reconstruction after upsampling
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
        )

        # High-frequency reconstruction
        self.high_freq_reconstruction = nn.Sequential(
            nn.Conv2d(base_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, num_channels, kernel_size=3, padding=1)
        )

    def check_size(self, x, scale):
        b, c, h, w = x.size()
        if h % scale != 0 or w % scale != 0:
            new_h = h - (h % scale)
            new_w = w - (w % scale)
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        # Ensure input size is divisible by scale factor
        x = self.check_size(x, self.scale_factor)

        # Extract and upsample high-frequency features
        high_freq_features = self.high_freq(x)
        high_freq_upscaled = self.upsample(high_freq_features)
        high_freq_out = self.high_freq_reconstruction(high_freq_upscaled)

        # Initial feature extraction
        x = self.initial(x)

        # Back-projection stages
        h_list = []
        for i in range(len(self.up_blocks)):
            h = self.up_blocks[i](x)
            h_list.append(h)
            if i < len(self.down_blocks):
                x = self.down_blocks[i](h)

        # Concatenate all high-resolution features
        out = torch.cat(h_list, dim=1)

        # Process features before upsampling
        out = self.pre_upsample(out)

        # Calculate expected output size
        expected_size = (x.size(2) * self.scale_factor, x.size(3) * self.scale_factor)

        # Upsample to final resolution
        out = F.interpolate(out, size=expected_size, mode='bilinear', align_corners=False)

        # Final convolution
        main_out = self.final_conv(out)

        # Ensure both outputs have the same size
        if main_out.size() != high_freq_out.size():
            main_out = F.interpolate(main_out, size=high_freq_out.size()[2:], mode='bilinear', align_corners=False)

        # Combine outputs
        out = main_out + high_freq_out * 0.1

        return out