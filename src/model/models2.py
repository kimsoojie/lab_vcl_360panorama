from util.base import *
import model.ops as ops

class BaseGenerator(nn.Module):
    def __init__(self):
        super(BaseGenerator, self).__init__()
        # -- Encoder
        # In 256 x 128
        self.block1 = ops.conv_norm_leak(64, 128, 3, 1, 1) # out 256 x 128
        self.block2 = ops.conv_norm_leak(128, 128, 4, 2, 1) # out 128 x 64
        # In 128 x 64
        self.block3 = ops.conv_norm_leak(128, 256, 3, 1, 1) # out 128 x 64
        self.block4 = ops.conv_norm_leak(256, 256, 4, 2, 1) # out 64 x 32
        # In 64 x 32
        self.block5 = ops.conv_norm_leak(256, 256, 3, 1, 1) # out 64 x 32
        self.block6 = ops.conv_norm_leak(256, 256, 4, 2, 1) # out 32 x 16
        # In 32 x 16
        self.block7 = ops.conv_norm_leak(256, 256, 3, 1, 1) # out 32 x 16
        self.block8 = ops.conv_norm_leak(256, 256, 4, 2, 1) # out 16 x 8
        # In 16 x 8
        self.block9 = ops.conv_norm_leak(256, 512, 3, 1, 1) # out 16 x 8
        self.block10 = ops.conv_norm_leak(512, 512, 4, 2, 1) # out 8 x 4
        # In 8 x 4
        self.block11 = ops.conv_norm_leak(512, 1024, 3, 1, 1) # out 8 x 4
        self.block12 = ops.conv_norm_leak(1024, 1024, 3, 1, 1) # out 8 x 4

        # -- Decoder
        # In 8 x 4
        self.dblock1 = ops.convT_norm_leak(1024, 512, 3, 1, 1) # out 8 x 4
        self.dblock2 = ops.convT_norm_leak(512 + 1024, 512, 3, 1, 1) # out 8 x 4
        self.dblock3 = ops.convT_norm_leak(512 + 512,  512, 4, 2, 1) # out 16 x 8
        # In 16 x 8
        self.dblock4 = ops.convT_norm_leak(512 + 512, 256, 3, 1, 1) # out 16 x 8
        self.dblock5 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1) # out 32 x 16
        # In 32 x 16
        self.dblock6 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1) # out 32 x 16
        self.dblock7 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1) # out 64 x 32
        # In 64 x 32
        self.dblock8 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1) # out 64 x 32
        self.dblock9 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1) # out 128 x 64
        # In 128 x 64
        self.dblock10 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1) # out 128 x 64
        self.dblock11 = ops.convT_norm_leak(256 + 128, 128, 4, 2, 1) # out 256 x 128
        # In 256 x 128
        self.dblock12 = ops.convT_norm_leak(128 + 128, 64, 3, 1, 1) # out  256 x 128


    def forward(self, x):
        enc1 = self.block1(x)
        enc2 = self.block2(enc1)
        enc3 = self.block3(enc2)
        enc4 = self.block4(enc3)
        enc5 = self.block5(enc4)
        enc6 = self.block6(enc5)
        enc7 = self.block7(enc6)
        enc8 = self.block8(enc7)
        enc9 = self.block9(enc8)
        enc10 = self.block10(enc9)
        enc11 = self.block11(enc10)
        enc12 = self.block12(enc11)

        dec1 = self.dblock1(enc12)
        dec2 = self.dblock2(ops.dstack(dec1,enc11))
        dec3 = self.dblock3(ops.dstack(dec2,enc10))
        dec4 = self.dblock4(ops.dstack(dec3,enc9))
        dec5 = self.dblock5(ops.dstack(dec4,enc8))
        dec6 = self.dblock6(ops.dstack(dec5,enc7))
        dec7 = self.dblock7(ops.dstack(dec6,enc6))
        dec8 = self.dblock8(ops.dstack(dec7,enc5))
        dec9 = self.dblock9(ops.dstack(dec8,enc4))
        dec10 = self.dblock10(ops.dstack(dec9,enc3))
        dec11 = self.dblock11(ops.dstack(dec10,enc2))
        dec12 = self.dblock12(ops.dstack(dec11,enc1))
        return dec12


class GeneratorSmall(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GeneratorSmall, self).__init__()
        # Bridge in
        self.bridge_in_small = ops.conv_leak(in_ch, 64, 3, 1, 1)
        # Small
        self.g_base = BaseGenerator()
        # Bridge out
        self.bridge_out_small = ops.convT_tanh(64, out_ch, 1, 1, 0)

    def forward(self, x):
        out = self.bridge_in_small(x)
        out = self.g_base(out)
        out = self.bridge_out_small(out)
        return out


class GeneratorMedium(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GeneratorMedium, self).__init__()
        # Bridge in
        self.bridge_in_medium = ops.conv_leak(in_ch, 64, 3, 1, 1)

        # Medium
        self.block_m1 = ops.conv_norm_leak(64, 64, 3, 1, 1) # out 512 x 256
        self.block_m2 = ops.conv_norm_leak(64, 64, 3, 1, 1) # out 512 x 256
        self.block_m3 = ops.conv_norm_leak(64, 64, 4, 2, 1) # out 256 x 128
        self.block_m4 = ops.conv_norm_leak(64, 64, 3, 1, 1) # out 256 x 128

        # Small
        self.bridge_in_small = ops.conv_leak(in_ch, 64, 3, 1, 1)
        self.g_base = BaseGenerator()
        self.bridge_out_small = ops.convT(64, out_ch, 1, 1, 0)

        # Medium
        self.dblock_m1 = ops.convT_norm_leak(64, 64, 3, 1, 1) # out 256 x 128
        self.dblock_m2 = ops.convT_norm_leak(64 + 64, 64, 4, 2, 1) # out 512 x 256
        self.dblock_m3 = ops.convT_norm_leak(64 + 64, 64, 3, 1, 1) # out 512 x 256
        self.dblock_m4 = ops.convT_norm_leak(64 + 64, 64, 3, 1, 1) # out 512 x 256

        # Bridge out
        self.bridge_out_medium = ops.convT(64, out_ch, 3, 1, 1)

    def forward(self, x):
        # Small
        in_small = self.bridge_in_small(ops.downsample(x,2))
        dec_small = self.g_base(in_small)
        out_small = self.bridge_out_small(dec_small)

        # Medium
        enc = self.bridge_in_medium(x)
        enc1 = self.block_m1(enc)
        enc2 = self.block_m2(enc1)
        enc3 = self.block_m3(enc2)
        enc4 = self.block_m4(enc3)
        dec1 = self.dblock_m1(enc4 + dec_small) # Add residual from small
        dec2 = self.dblock_m2(ops.dstack(dec1,enc3))
        dec3 = self.dblock_m3(ops.dstack(dec2,enc2))
        dec4 = self.dblock_m4(ops.dstack(dec3,enc1))
        out_medium = self.bridge_out_medium(dec4)
        out_medium = F.tanh(ops.upsample(out_small,2) + out_medium) # Add residual form small

        return [out_small, out_medium]

class GeneratorLarge(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(GeneratorLarge, self).__init__()
        # Bridge in
        self.bridge_in_large = ops.conv_leak(in_ch, 64, 3, 1, 1)

        # Large
        self.block_l1 = ops.conv_norm_leak(64, 64, 3, 1, 1) # out 1024 x 512
        self.block_l2 = ops.conv_norm_leak(64, 64, 3, 1, 1) # out 1024 x 512
        self.block_l3 = ops.conv_norm_leak(64, 64, 4, 2, 1) # out 512 x 256
        self.block_l4 = ops.conv_norm_leak(64, 64, 3, 1, 1) # out 512 x 256

        # Medium
        self.bridge_in_medium = ops.conv_leak(in_ch, 64, 3, 1, 1)
        self.block_m1 = ops.conv_norm_leak(64, 64, 3, 1, 1) # out 512 x 256
        self.block_m2 = ops.conv_norm_leak(64, 64, 3, 1, 1) # out 512 x 256
        self.block_m3 = ops.conv_norm_leak(64, 64, 4, 2, 1) # out 256 x 128
        self.block_m4 = ops.conv_norm_leak(64, 64, 3, 1, 1) # out 256 x 128

        # Small
        self.bridge_in_small = ops.conv_leak(in_ch, 64, 3, 1, 1)
        self.g_base = BaseGenerator()
        self.bridge_out_small = ops.convT(64, out_ch, 1, 1, 0)

        # Medium
        self.dblock_m1 = ops.convT_norm_leak(64, 64, 3, 1, 1) # out 256 x 128
        self.dblock_m2 = ops.convT_norm_leak(64 + 64, 64, 4, 2, 1) # out 512 x 256
        self.dblock_m3 = ops.convT_norm_leak(64 + 64, 64, 3, 1, 1) # out 512 x 256
        self.dblock_m4 = ops.convT_norm_leak(64 + 64, 64, 3, 1, 1) # out 512 x 256
        self.bridge_out_medium = ops.convT(64, out_ch, 3, 1, 1)

        # Large
        self.dblock_l1 = ops.convT_norm_leak(64, 64, 3, 1, 1) # out 512 x 256
        self.dblock_l2 = ops.convT_norm_leak(64 + 64, 64, 4, 2, 1) # out 1024 x 512
        self.dblock_l3 = ops.convT_norm_leak(64 + 64, 64, 3, 1, 1) # out 1024 x 512
        self.dblock_l4 = ops.convT_norm_leak(64 + 64, 64, 3, 1, 1) # out 1024 x 512

        # Bridge out
        self.bridge_out_large = ops.convT(64, 3, 5, 1, 2)
        # self.bridge_out_large = ops.convT(64, 3, 3, 1, 1)


    def forward(self, x):
        # Small
        in_small = self.bridge_in_small(ops.downsample(x, 4))
        dec_small = self.g_base(in_small)
        out_small = self.bridge_out_small(dec_small)

        # Medium
        encm = self.bridge_in_medium(ops.downsample(x, 2))
        enc1m = self.block_m1(encm)
        enc2m = self.block_m2(enc1m)
        enc3m = self.block_m3(enc2m)
        enc4m = self.block_m4(enc3m)
        dec1m = self.dblock_m1(enc4m + dec_small) # Add residual from small
        dec2m = self.dblock_m2(ops.dstack(dec1m,enc3m))
        dec3m = self.dblock_m3(ops.dstack(dec2m,enc2m))
        dec_medium = self.dblock_m4(ops.dstack(dec3m,enc1m))
        out_medium = self.bridge_out_medium(dec_medium)
        out_medium = ops.upsample(out_small,2) + out_medium # Add residual form small

        # Large
        enc = self.bridge_in_large(x)
        enc1 = self.block_l1(enc)
        enc2 = self.block_l2(enc1)
        enc3 = self.block_l3(enc2)
        enc4 = self.block_l4(enc3)
        dec1 = self.dblock_l1(enc4 + dec_medium) # Add residual from medium
        dec2 = self.dblock_l2(ops.dstack(dec1,enc3))
        dec3 = self.dblock_l3(ops.dstack(dec2,enc2))
        dec_large = self.dblock_l4(ops.dstack(dec3,enc1))
        out_large = self.bridge_out_large(dec_large)
        out_large = F.tanh(ops.upsample(out_medium,2) + out_large) # Add residual from medium

        return [out_small, out_medium, out_large]


class BaseDiscriminator(nn.Module):
    def __init__(self):
        super(BaseDiscriminator, self).__init__()
        self.d_base = nn.Sequential(
            ops.conv_leak(6, 64, 4, 2, 1),
            ops.conv_norm_leak(64, 128, 4, 2, 1),
            ops.conv_norm_leak(128, 256, 4, 2, 1),
            ops.conv_norm_leak(256, 512, 4, 1, 1),
            ops.conv_sigmoid(512, 1, 4, 1, 1),
        )

    def forward(self, x):
        out = self.d_base(x)
        return out

class DiscriminatorSmall(nn.Module):
    def __init__(self):
        super(DiscriminatorSmall, self).__init__()
        self.d_small = BaseDiscriminator()

    def forward(self, x):
        out = self.d_small(x)
        return out

class DiscriminatorMedium(nn.Module):
    def __init__(self):
        super(DiscriminatorMedium, self).__init__()
        self.d_medium = nn.Sequential(
            ops.conv_leak(6, 64, 4, 2, 1),
            ops.conv_norm_leak(64, 128, 4, 2, 1),
            ops.conv_norm_leak(128, 256, 4, 2, 1),
            ops.conv_norm_leak(256, 512, 4, 2, 1),
            ops.conv_sigmoid(512, 1, 4, 2, 1),
        )

    def forward(self, x_medium):
        out_medium = self.d_medium(x_medium)

        return out_medium

class DiscriminatorLarge(nn.Module):
    def __init__(self):
        super(DiscriminatorLarge, self).__init__()
        self.d_large = nn.Sequential(
            ops.conv_leak(6, 64, 4, 2, 1),
            ops.conv_norm_leak(64, 128, 4, 2, 1),
            ops.conv_norm_leak(128, 256, 4, 2, 1),
            ops.conv_norm_leak(256, 256, 4, 2, 1),
            ops.conv_norm_leak(256, 512, 4, 2, 1),
            ops.conv_sigmoid(512, 1, 4, 2, 1),
        )

    def forward(self, x_large):
        out_large = self.d_large(x_large)
        return out_large


class SemanticNet(nn.Module):
    def __init__(self, in_ch, depth=2):
        super(SemanticNet, self).__init__()
        self.depth = depth
        self.conv1 = nn.Conv2d(in_ch, 100, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = []
        self.bn2 = []
        for i in range(depth - 1):
            self.conv2.append(nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(100))
        self.conv3 = nn.Conv2d(100, 100, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(100)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(self.depth - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x



