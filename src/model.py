import torch
import torch.nn as nn
import timm


class ConvBlock(nn.Module):
    """Standard Convolutional Block (Conv -> BN -> ReLU)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """Up-sampling followed by a convolutional block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TransUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        # --- Transformer Encoder ---
        self.vit_encoder = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )

        # The encoder is now UN-FROZEN by default, allowing it to be fine-tuned.

        self.vit_output_dim = self.vit_encoder.embed_dim
        self.reshape_conv = nn.Conv2d(self.vit_output_dim, 512, kernel_size=1)

        # --- CNN Decoder (U-Net style) ---
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.up1 = UpConv(512 + 128, 256)
        self.up2 = UpConv(256 + 64, 128)
        self.out_conv = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, x):
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))

        x_vit = torch.nn.functional.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )
        vit_out = self.vit_encoder.forward_features(x_vit)

        vit_out = vit_out[:, 1:, :]
        batch_size, _, embed_dim = vit_out.shape
        h, w = 14, 14
        vit_out = vit_out.permute(0, 2, 1).reshape(batch_size, embed_dim, h, w)

        vit_out_upscaled = torch.nn.functional.interpolate(
            vit_out, size=(48, 48), mode="bilinear"
        )
        vit_out_reshaped = self.reshape_conv(vit_out_upscaled)

        dec1 = self.up1(vit_out_reshaped, skip2)
        dec2 = self.up2(dec1, skip1)

        logits = self.out_conv(dec2)
        return torch.sigmoid(logits)
