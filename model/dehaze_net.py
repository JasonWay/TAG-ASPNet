import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=bias,
    )


# ----- Multi-dimensional attention blocks (channel/spatial/pixel) -----
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        return self.sa(x2)


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, max(dim // reduction, 4), 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // reduction, 4), dim, 1, bias=True),
        )

    def forward(self, x):
        return self.ca(self.gap(x))


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa2 = nn.Conv2d(
            2 * dim, dim, 7, padding=3, padding_mode="reflect", groups=dim, bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        x_u = x.unsqueeze(2)
        p_u = pattn1.unsqueeze(2)
        x2 = torch.cat([x_u, p_u], dim=2).flatten(1, 2)
        return self.sigmoid(self.pa2(x2))


class TRARB(nn.Module):
    """TriAttentionResidualBlock: residual block with channel/spatial-guided pixel gating."""

    def __init__(self, conv, dim, kernel_size, reduction=8):
        super().__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2
        return res + x


class RRB(nn.Module):
    """ResidualRefineBlock: two-conv residual refinement block for shallow stages."""

    def __init__(self, conv, dim, kernel_size):
        super().__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res) + x
        return res


class GSF(nn.Module):
    """GatedSkipFusion: adaptive gated fusion for encoder skip and upsampled branch."""

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        return self.conv(result)


class LKDRB(nn.Module):
    """LargeKernelDWRefineBlock: large-kernel depthwise refinement + 1x1 projection."""

    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(
            dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode="reflect", bias=True
        )
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.pw(self.act(self.dw(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class DSFB(nn.Module):
    """DecoderSkipFusionBlock: gated fusion of upsampled decoder feature and encoder skip."""

    def __init__(self, low_ch, skip_ch, out_ch, reduction=8):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(low_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, 1, bias=True)
        red = max(4, min(8, out_ch // 4))
        self.cga = GSF(out_ch, reduction=red)

    def forward(self, low, skip):
        u = self.up(low)
        s = self.skip_proj(skip)
        if u.shape[-2:] != s.shape[-2:]:
            # Handle odd-sized inputs: transposed conv can differ by 1px from skip branch.
            u = F.interpolate(u, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return self.cga(s, u)


class TAGASPNet(nn.Module):
    """
    TAG-ASPNet: A Tri-Attention Gated Dehazing Network with Atmospheric Scattering Prior.
    Shared-encoder multi-head dehazing network.
    Backbone: shallow residual refinement + tri-attention bottleneck + gated skip fusion decoder.
    Includes 5x5 depthwise refinement in bottleneck to enlarge receptive field.
    Outputs:
      J: dehazed image in [0, 1]
      t: transmission map in [0.1, 1.0]
      A: atmosphere map in [0.75, 0.95]
    """

    def __init__(self, base_channels=32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        conv = default_conv

        self.stem = nn.Sequential(
            conv(3, c1, 3, bias=True),
            nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(
            RRB(conv, c1, 3),
            RRB(conv, c1, 3),
            RRB(conv, c1, 3),
            RRB(conv, c1, 3),
        )
        self.down1 = nn.Sequential(
            conv(c1, c2, 3, stride=2, bias=True),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            RRB(conv, c2, 3),
            RRB(conv, c2, 3),
            RRB(conv, c2, 3),
            RRB(conv, c2, 3),
        )
        self.down2 = nn.Sequential(
            conv(c2, c3, 3, stride=2, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fe_bottleneck = conv(c3, c3, 3, bias=True)

        self.bottleneck = nn.Sequential(
            TRARB(conv, c3, 3),
            TRARB(conv, c3, 3),
            TRARB(conv, c3, 3),
            LKDRB(c3),
            TRARB(conv, c3, 3),
            TRARB(conv, c3, 3),
            TRARB(conv, c3, 3),
            LKDRB(c3),
            TRARB(conv, c3, 3),
            TRARB(conv, c3, 3),
            SEBlock(c3),
        )

        self.dec2 = DSFB(c3, c2, c2)
        self.dec1 = DSFB(c2, c1, c1)

        self.post_dec2 = nn.Sequential(
            RRB(conv, c2, 3),
            RRB(conv, c2, 3),
        )
        self.post_dec1 = nn.Sequential(
            RRB(conv, c1, 3),
            RRB(conv, c1, 3),
        )

        self.j_head = nn.Sequential(
            ConvBlock(c1, c1, kernel_size=3),
            nn.Conv2d(c1, 3, kernel_size=3, padding=1),
        )
        self.t_head = nn.Sequential(
            ConvBlock(c1, c1, kernel_size=3),
            nn.Conv2d(c1, 1, kernel_size=1),
        )
        self.a_head = nn.Sequential(
            ConvBlock(c3, c2, kernel_size=3),
            nn.Conv2d(c2, 3, kernel_size=1),
        )

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(self.down1(s1))
        b0 = self.fe_bottleneck(self.down2(s2))
        b = self.bottleneck(b0)

        d2 = self.post_dec2(self.dec2(b, s2))
        d1 = self.post_dec1(self.dec1(d2, s1))

        j_res = self.j_head(d1)
        J = torch.clamp(x + j_res, 0.0, 1.0)

        t_logits = self.t_head(d1)
        t = 0.1 + 0.9 * torch.sigmoid(t_logits)

        a_lowres = self.a_head(b)
        a_map = F.interpolate(a_lowres, size=x.shape[-2:], mode="bilinear", align_corners=False)
        A = 0.75 + 0.2 * torch.sigmoid(a_map)

        return J, t, A
