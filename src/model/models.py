from util.base import *
import model.ops as ops
from util.opt import Options

opt = Options(sys.argv[0])

"""
 Generator Model
 =================
"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # encoder
        # self.block1 = ops.conv_leak(3, 128, (4,6), (2,2), (1,2))
        self.block1 = ops.conv_relu(3, 128, (4,4), (2,2), (1,1))
        self.block2 = ops.conv_norm_relu(128, 256, (4,4), (2,2), (1,1))
        self.block3 = ops.conv_norm_relu(256, 512, (4,4), (2,2), (1,1))
        self.block4 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1))
        self.block5 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1))
        self.block6 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1))
        self.block7 = ops.conv_norm_relu(512, 1024, (4,4), (2,2), (1,1))
        # decoder
        self.dblock1 = ops.convT_norm_leak(1024, 512, (4,4), (2,2), (1,1))
        self.dblock2 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock3 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock4 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock5 = ops.convT_norm_leak(512 + 512, 256, (4,4), (2,2), (1,1))
        self.dblock6 = ops.convT_norm_leak(256 + 256, 128, (4,4), (2,2), (1,1))
        self.dblock7 = ops.convT(128 + 128, 3, (4,4), (2,2), (1,1))

    def forward(self, x):
        enc1 = self.block1(x)
        enc2 = self.block2(enc1)
        enc3 = self.block3(enc2)
        enc4 = self.block4(enc3)
        enc5 = self.block5(enc4)
        enc6 = self.block6(enc5)
        enc7 = self.block7(enc6)
        # Decoder
        dec1 = self.dblock1(enc7)
        dec2 = self.dblock2(ops.dstack(dec1,enc6))
        dec3 = self.dblock3(ops.dstack(dec2,enc5))
        dec4 = self.dblock4(ops.dstack(dec3,enc4))
        dec5 = self.dblock5(ops.dstack(dec4,enc3))
        dec6 = self.dblock6(ops.dstack(dec5,enc2))
        dec7 = self.dblock7(ops.dstack(dec6,enc1))

        return F.tanh(dec7)


"""
 Progressive Generator 3x Scale 
 =================
"""
class GeneratorSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(GeneratorSmall, self).__init__()
        norm = 'instnorm'
        self.block1 = ops.conv_relu(in_ch, 128, (4,4), (2,2), (1,1))
        self.block2 = ops.conv_norm_relu(128, 256, (4,4), (2,2), (1,1))
        self.block3 = ops.conv_norm_relu(256, 512, (4,4), (2,2), (1,1))
        self.block4 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1))
        self.block5 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1))
        self.block6 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1))
        self.block7 = ops.conv_norm_relu(512, 1024, (4,4), (2,2), (1,1))
        self.block8 = ops.conv_norm_relu(1024, 1024, (4,4), (2,2), (1,1))

        # decoder
        self.dblock1 = ops.convT_norm_leak(1024, 1024, (4,4), (2,2), (1,1))
        self.dblock2 = ops.convT_norm_leak(1024 + 1024, 512, (4,4), (2,2), (1,1))
        self.dblock3 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock4 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock5 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock6 = ops.convT_norm_leak(512 + 512, 256, (4,4), (2,2), (1,1))
        self.rgb_small = ops.convT(256, out_ch, 1, 1, 0)

    def forward(self, x):
        enc1 = self.block1(x)
        enc2 = self.block2(enc1)
        enc3 = self.block3(enc2)
        enc4 = self.block4(enc3)
        enc5 = self.block5(enc4)
        enc6 = self.block6(enc5)
        enc7 = self.block7(enc6)
        enc8 = self.block8(enc7)

        dec1 = self.dblock1(enc8)
        dec2 = self.dblock2(ops.dstack(dec1, enc7))
        dec3 = self.dblock3(ops.dstack(dec2, enc6))
        dec4 = self.dblock4(ops.dstack(dec3, enc5))
        dec5 = self.dblock5(ops.dstack(dec4, enc4))
        dec6 = self.dblock6(ops.dstack(dec5, enc3))
        dec_out = F.tanh(self.rgb_small(dec6))

        return dec_out

class GeneratorMedium(nn.Module):
    def __init__(self):
        super(GeneratorMedium, self).__init__()
        norm = 'instnorm'
        self.block1 = ops.conv_relu(3, 128, (4,4), (2,2), (1,1))
        self.block2 = ops.conv_norm_relu(128, 256, (4,4), (2,2), (1,1), norm=norm)
        self.block3 = ops.conv_norm_relu(256, 512, (4,4), (2,2), (1,1), norm=norm)
        self.block4 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1), norm=norm)
        self.block5 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1), norm=norm)
        self.block6 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1), norm=norm)
        self.block7 = ops.conv_norm_relu(512, 1024, (4,4), (2,2), (1,1), norm=norm)
        self.block8 = ops.conv_norm_relu(1024, 1024, (4,4), (2,2), (1,1), norm=norm)

        # decoder
        self.dblock1 = ops.convT_norm_leak(1024, 1024, (4,4), (2,2), (1,1), norm=norm)
        self.dblock2 = ops.convT_norm_leak(1024 + 1024, 512, (4,4), (2,2), (1,1), norm=norm)
        self.dblock3 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1), norm=norm)
        self.dblock4 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1), norm=norm)
        self.dblock5 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1), norm=norm)
        self.dblock6 = ops.convT_norm_leak(512 + 512, 256, (4,4), (2,2), (1,1), norm=norm)
        self.dblock7 = ops.convT_norm_leak(256 + 256, 128, (4,4), (2,2), (1,1), norm=norm)
        self.dblock8 = ops.convT_norm_leak(128, 128, (3,3), (1,1), (1,1), norm=norm)
        self.dblock9 = ops.convT_norm_leak(128, 128, (3,3), (1,1), (1,1), norm=norm)
        self.dblock10 = ops.convT_norm_leak(128, 128, (3,3), (1,1), (1,1), norm=norm)
        self.dblock11 = ops.convT_norm_leak(128, 128, (3,3), (1,1), (1,1), norm=norm)
        self.rgb_small = ops.convT(256, 3, 1, 1, 0)
        self.rgb_medium = ops.convT(128, 3, 3, 1, 1)

    def forward(self, x):
        enc1 = self.block1(x)
        enc2 = self.block2(enc1)
        enc3 = self.block3(enc2)
        enc4 = self.block4(enc3)
        enc5 = self.block5(enc4)
        enc6 = self.block6(enc5)
        enc7 = self.block7(enc6)
        enc8 = self.block8(enc7)

        dec1 = self.dblock1(enc8)
        dec2 = self.dblock2(ops.dstack(dec1, enc7))
        dec3 = self.dblock3(ops.dstack(dec2, enc6))
        dec4 = self.dblock4(ops.dstack(dec3, enc5))
        dec5 = self.dblock5(ops.dstack(dec4, enc4))
        dec6 = self.dblock6(ops.dstack(dec5, enc3))
        dec7 = self.dblock7(ops.dstack(dec6, enc2))
        dec8 = self.dblock8(dec7)
        dec9 = self.dblock9(dec8)
        dec10 = self.dblock10(dec9)
        dec11 = self.dblock11(dec10)
        # out_small = self.rgb_small(dec6)
        out_medium = self.rgb_medium(dec11)
        # dec_out = ops.upsample(out_small, 2, mode='bilinear') + out_medium

        return out_medium, torch.tanh(out_medium)

class GeneratorMedium2(nn.Module):
    def __init__(self):
        super(GeneratorMedium2, self).__init__()
        self.block1 = ops.conv_relu(3, 128, (4,4), (2,2), (1,1))
        self.block2 = ops.conv_norm_relu(128, 256, (4,4), (2,2), (1,1))
        self.block3 = ops.conv_norm_relu(256, 512, (4,4), (2,2), (1,1))
        self.block4 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1))
        self.block5 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1))
        self.block6 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1))
        self.block7 = ops.conv_norm_relu(512, 1024, (4,4), (2,2), (1,1))
        self.block8 = ops.conv_norm_relu(1024, 1024, (4,4), (2,2), (1,1))

        # decoder
        self.dblock1 = ops.convT_norm_leak(1024, 1024, (4,4), (2,2), (1,1))
        self.dblock2 = ops.convT_norm_leak(1024 + 1024, 512, (4,4), (2,2), (1,1))
        self.dblock3 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock4 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock5 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock6 = ops.convT_norm_leak(512 + 512, 256, (4,4), (2,2), (1,1))
        self.rgb_small = ops.convT(256, 3, 1, 1, 0)
        # self.rgb_medium = ops.convT(256, 3, 4, 2, 1)


    def forward(self, x):
        enc1 = self.block1(x)
        enc2 = self.block2(enc1)
        enc3 = self.block3(enc2)
        enc4 = self.block4(enc3)
        enc5 = self.block5(enc4)
        enc6 = self.block6(enc5)
        enc7 = self.block7(enc6)
        enc8 = self.block8(enc7)
        enc9 = self.block8(enc8)

        dec1 = self.dblock1(enc9)
        dec2 = self.dblock2(ops.dstack(dec1, enc8))
        dec3 = self.dblock3(ops.dstack(dec2, enc7))
        dec4 = self.dblock4(ops.dstack(dec3, enc6))
        dec5 = self.dblock5(ops.dstack(dec4, enc5))
        dec6 = self.dblock6(ops.dstack(dec5, enc4))
        dec7 = self.dblock7(ops.dstack(dec6, enc3))
        dec8 = self.dblock8(ops.dstack(dec7, enc2))
        # out_small = self.rgb_small(dec6)
        out_medium = self.rgb_medium(dec8 + x)
        # dec_out = ops.upsample(out_small, 2, mode='bilinear') + out_medium

        return F.tanh(out_medium), F.tanh(out_medium)

class GeneratorLarge(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GeneratorLarge, self).__init__()
        self.block1 = ops.conv_relu(in_ch, 128, (4,4), (2,2), (1,1)) # 256 x 512
        self.block2 = ops.conv_norm_relu(128, 256, (4,4), (2,2), (1,1)) # 128 x 256
        self.block3 = ops.conv_norm_relu(256, 512, (4,4), (2,2), (1,1)) # 64 x 128
        self.block4 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1)) # 32 x 64
        self.block5 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1)) # 16 x 32
        self.block6 = ops.conv_norm_relu(512, 512, (4,4), (2,2), (1,1)) # 8 x 16
        self.block7 = ops.conv_norm_relu(512, 1024, (4,4), (2,2), (1,1)) # 4 x 8
        self.block8 = ops.conv_norm_relu(1024, 1024, (4,4), (2,2), (1,1)) # 2 x 4

        # decoder
        self.dblock1 = ops.convT_norm_leak(1024, 1024, (4,4), (2,2), (1,1))
        self.dblock2 = ops.convT_norm_leak(1024 + 1024, 512, (4,4), (2,2), (1,1))
        self.dblock3 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock4 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock5 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock6 = ops.convT_norm_leak(512 + 512, 256, (4,4), (2,2), (1,1))
        self.dblock7 = ops.convT_norm_leak(256 + 256, 128, (4,4), (2,2), (1,1))
        self.dblock8 = ops.convT(128 + 128, out_ch, (4,4), (2,2), (1,1))
        self.rgb_small = ops.convT(256, 3, 1, 1, 0)
        self.rgb_medium = ops.convT(128, 3, 1, 1, 0)

    def forward(self, x):
        enc1 = self.block1(x)
        enc2 = self.block2(enc1)
        enc3 = self.block3(enc2)
        enc4 = self.block4(enc3)
        enc5 = self.block5(enc4)
        enc6 = self.block6(enc5)
        enc7 = self.block7(enc6)
        enc8 = self.block8(enc7)

        dec1 = self.dblock1(enc8)
        dec2 = self.dblock2(ops.dstack(dec1, enc7))
        dec3 = self.dblock3(ops.dstack(dec2, enc6))
        dec4 = self.dblock4(ops.dstack(dec3, enc5))
        dec5 = self.dblock5(ops.dstack(dec4, enc4))
        dec6 = self.dblock6(ops.dstack(dec5, enc3))
        dec7 = self.dblock7(ops.dstack(dec6, enc2))
        dec8 = self.dblock8(ops.dstack(dec7, enc1))
        out_small = self.rgb_small(dec6)
        out_med = self.rgb_medium(dec7)
        out_med = ops.upsample(out_small, 2, mode='bilinear') + out_med
        out_large = F.tanh(ops.upsample(out_med, 2, mode='bilinear') + dec8)

        return [out_small, out_med, out_large]

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.convx = nn.Conv2d(3,1,3,1,1,bias=False)
        self.convy = nn.Conv2d(3,1,3,1,1,bias=False)

        kx = torch.tensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        ky = torch.tensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
        kx = kx.expand((1,3,3,3)).type(torch.FloatTensor)
        ky = ky.expand((1,3,3,3)).type(torch.FloatTensor)

        with torch.no_grad():
            self.convx.weight = nn.Parameter(kx)
            self.convy.weight = nn.Parameter(ky)

    def forward(self, inputs):
        outx  = self.convx(inputs)
        outy  = self.convy(inputs)
        return [outx, outy]


"""
 Multi Scale Discriminator
 =========================
    multi resolution discriminator using base class of 
    Discriminator. Downscale input by factor 2 and 4.
"""
class MultiDiscriminator(nn.Module):
    def __init__(self):
        super(MultiDiscriminator, self).__init__()
        self.D_high = Discriminator()
        self.D_med = Discriminator()
        self.D_low = Discriminator()

    def forward(self, x):
        # High
        out_high = self.D_high(x)
        # Med
        x = ops.downsample(x)
        out_med = self.D_med(x)
        # Low
        x = ops.downsample(x)
        out_low = self.D_low(x)

        return [out_high, out_med, out_low]


"""
 Single Discriminator
 ======================
"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            ops.conv_leak(6, 64, 4, 2, 1),
            ops.conv_norm_leak(64, 128, 4, 2, 1),
            ops.conv_norm_leak(128, 256, 4, 2, 1),
            ops.conv_norm_leak(256, 512, 4, 1, 1),
            ops.conv_sigmoid(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.discriminator(x)


"""
 Progressive Discriminator 3x Scale
 ======================
"""
class DiscriminatorSmall(nn.Module):
    def __init__(self):
        super(DiscriminatorSmall, self).__init__()
        self.d_small = nn.Sequential(
            ops.conv_leak(6, 64, 4, 2, 1),
            ops.conv_norm_leak(64, 128, 4, 2, 1),
            ops.conv_norm_leak(128, 256, 4, 2, 1),
            ops.conv_norm_leak(256, 512, 4, 1, 1),
            nn.Dropout(0.5),
            ops.conv_sigmoid(512, 1, 4, 1, 1),
        )

    def forward(self, x):
        out = self.d_small(x)
        return out

class DiscriminatorMedium(nn.Module):
    def __init__(self):
        super(DiscriminatorMedium, self).__init__()
        self.d_small = nn.Sequential(
            ops.conv_leak(6, 64, 4, 2, 1),
            ops.conv_norm_leak(64, 128, 4, 2, 1),
            ops.conv_norm_leak(128, 256, 4, 2, 1),
            ops.conv_norm_leak(256, 512, 4, 1, 1)
        )
        self.d_med = nn.Sequential(
            ops.conv_norm_leak(512, 512, 4, 1, 1),
            ops.conv_sigmoid(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        out = self.d_small(x)
        out = self.d_med(out)
        return out

class DiscriminatorLarge(nn.Module):
    def __init__(self):
        super(DiscriminatorLarge, self).__init__()
        self.d_small = nn.Sequential(
            ops.conv_leak(6, 64, 4, 2, 1),
            ops.conv_norm_leak(64, 128, 4, 2, 1),
            ops.conv_norm_leak(128, 256, 4, 2, 1),
            ops.conv_norm_leak(256, 512, 4, 1, 1)
        )
        self.d_med = nn.Sequential(
            ops.conv_norm_leak(512, 512, 4, 1, 1)
        )
        self.d_large = nn.Sequential(
            ops.conv_norm_leak(512, 512, 4, 1, 1),
            ops.conv_sigmoid(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        out = self.d_small(x)
        out = self.d_med(out)
        out = self.d_large(out)
        return out



"""
 Generator with Segmentation
 ======================
"""
class GeneratorSmallSegment(nn.Module):
    def __init__(self, num_class):
        super(GeneratorSmallSegment, self).__init__()
        self.recon_small = GeneratorSmall(in_ch=3+num_class, out_ch=3)
        self.segment_small = Unet(in_ch=3, out_ch=3+num_class)

    def forward(self, x):
        out = self.segment_small(x)
        out_img = out[0]
        in_segment = out[1]
        out_recon = self.recon_small(ops.dstack(out_img, in_segment))
        return out_recon , in_segment, out_img

class GeneratorLargeSegment(nn.Module):
    def __init__(self):
        super(GeneratorLargeSegment, self).__init__()
        self.recon_large = GeneratorLarge(in_ch=3, out_ch=3)

    def forward(self, x):
        out = self.recon_large(x)
        return out



"""
 Wnet Segmentation
 ======================
"""
class Unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(Unet, self).__init__()
        self.block1 = ops.conv_leak(in_ch, 64, (4,4), (2,2), (1,1))
        self.block2 = ops.conv_norm_leak(64, 128, (4,4), (2,2), (1,1))
        self.block3 = ops.conv_norm_leak(128, 256, (4,4), (2,2), (1,1))
        self.block4 = ops.conv_norm_leak(256, 512, (4,4), (2,2), (1,1))
        self.block5 = ops.conv_norm_leak(512, 512, (4,4), (2,2), (1,1))
        self.block6 = ops.conv_norm_leak(512, 512, (4,4), (2,2), (1,1))

        # decoder
        self.dblock1 = ops.convT_norm_leak(512, 512, (4,4), (2,2), (1,1))
        self.dblock2 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock3 = ops.convT_norm_leak(512 + 512, 512, (4,4), (2,2), (1,1))
        self.dblock4 = ops.convT_norm_leak(256 + 512, 256, (4,4), (2,2), (1,1))
        self.dblock5 = ops.convT_norm_leak(128 + 256, 128, (4,4), (2,2), (1,1))
        self.dblock6 = ops.convT(64 + 128, out_ch, (4,4), (2,2), (1,1))

    def forward(self, x):
        enc1 = self.block1(x)
        enc2 = self.block2(enc1)
        enc3 = self.block3(enc2)
        enc4 = self.block4(enc3)
        enc5 = self.block5(enc4)
        enc6 = self.block6(enc5)

        dec1 = self.dblock1(enc6)
        dec2 = self.dblock2(ops.dstack(dec1, enc5))
        dec3 = self.dblock3(ops.dstack(dec2, enc4))
        dec4 = self.dblock4(ops.dstack(dec3, enc3))
        dec5 = self.dblock5(ops.dstack(dec4, enc2))
        dec6 = self.dblock6(ops.dstack(dec5, enc1))

        out_pix = F.tanh(dec6[:,0:3,:,:])
        out_seg = F.softmax(dec6[:,3:,:,:], dim=1)

        return [out_pix, out_seg]

class Wnet(nn.Module):
    def __init__(self, num_class):
        super(Wnet, self).__init__()
        self.segment = Unet(3, num_class)
        self.recons = Unet(num_class, 3)

    def forward(self, x):
        out_segment = F.relu(self.segment(x))
        out_segment = F.softmax(out_segment, dim=1)
        out_recons = F.tanh(self.recons(out_segment))
        return out_recons, out_segment

"""
 FOV Network
 ======================
"""
class FovNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FovNet, self).__init__()
        self.conv_block = nn.Sequential(
            ops.conv_leak(in_ch, 16, 4, 2, 1),
            ops.conv_norm_leak(16, 32, 4, 2, 1),
            ops.conv_norm_leak(32, 64, 4, 2, 1),
            ops.conv_norm_leak(64, 128, 4, 2, 1),
            ops.conv_norm_leak(128, 256, 4, 2, 1),
            ops.conv_norm_leak(256, 512, 4, 2, 1),
        )
        self.fc1 = nn.Linear(4 * 16 * 512, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 256)

    def forward(self, x):
        conv_out = self.conv_block(x)
        flat = torch.view(conv_out, -1)
        out = F.leaky_relu(self.fc1(flat))
        out = F.leaky_relu(self.fc2(out))
        out = F.softmax(self.fc3(out))
        out_max = torch.argmax(out, dim=-1)
        return out, out_max

"""
 VGG Network Loss
 ======================
"""
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_net = models.vgg16(pretrained=False).features
        weights = torch.load(opt.model_path + '/vgg16-397923af.pth')
        vgg_net.load_state_dict(weights, strict=False)
        self.feat_layers = vgg_net
        self.class_layers = nn.Sequential(
            nn.Linear(512 * 4 * 16, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 128),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.feat_layers(x)
        out = out.view(-1, 512 * 4 * 16)
        out = self.class_layers(out)
        return out

class FOVnetwork(nn.Module):
    def __init__(self):
        super(FOVnetwork, self).__init__()
        self.soft_argmax = SoftArgmax1D()
        self.conv0_0 = nn.Sequential(
            ops.conv_relu(3, 64, 3, 1, 1),
            ops.conv_norm_relu(64, 64, 4, 2, 1),
            ops.conv_norm_relu(64, 64, 4, 2, 1),
        )
        self.conv0_1 = nn.Sequential(
            ops.conv_relu(3, 64, 3, 1, 1),
            ops.conv_norm_relu(64, 64, 4, 2, 1),
            ops.conv_norm_relu(64, 64, 4, 2, 1),
        )
        self.conv0_2 = nn.Sequential(
            ops.conv_relu(3, 64, 3, 1, 1),
            ops.conv_norm_relu(64, 64, 4, 2, 1),
            ops.conv_norm_relu(64, 64, 4, 2, 1),
        )
        self.conv0_3 = nn.Sequential(
            ops.conv_relu(3, 64, 3, 1, 1),
            ops.conv_norm_relu(64, 64, 4, 2, 1),
            ops.conv_norm_relu(64, 64, 4, 2, 1),
        )

        self.conv1 = nn.Sequential(
            ops.conv_norm_relu(64, 128, 4, 2, 1),
            ops.conv_norm_relu(128, 256, 4, 2, 1),
            ops.conv_norm_relu(256, 256, 4, 2, 1),
            ops.conv_norm_relu(256, 256, 4, 2, 1),
        )

        self.lin0 = nn.Sequential(
            nn.Linear(256 * 2 * 8, 2048),
            # nn.Dropout(0.0),
            # nn.ReLU(True),
            nn.Linear(2048, 128),
            nn.Sigmoid(),
        )
        self.lin1 = nn.Sequential(
            nn.Linear(128, 2048),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(2048, 256 * 2 * 8),
            nn.Dropout(),
            nn.ReLU(True),
        )

        self.deconv0 = nn.Sequential(
            ops.convT_norm_relu(256, 256, 4, 2, 1),
            ops.convT_norm_relu(256, 256, 4, 2, 1),
            ops.convT_norm_relu(256, 128, 4, 2, 1),
            ops.convT_norm_relu(128, 128, 4, 2, 1),
            ops.convT_norm_relu(128, 64, 4, 2, 1),
            ops.convT(64, 3, 4, 2, 1),
        )


    def forward(self, x):
        out0 = self.conv0_0(x[:,:,:,0:128])
        out1 = self.conv0_1(x[:,:,:,128:256])
        out2 = self.conv0_2(x[:,:,:,256:384])
        out3 = self.conv0_3(x[:,:,:,384:512])

        out_cat = torch.cat((out0, out1, out2, out3), 3) # horiz cat
        out = self.conv1(out_cat)
        print('out_size',out.size())
        out_fc = self.lin0(out.view(-1, 256 * 2 * 8))

        # idx_tensor = self.soft_argmax(out_fc)

        out = self.lin1(out_fc)
        out = self.deconv0(out.view(-1, 256, 2 , 8))
        print(out.size())
        out = F.tanh(out)
        return out, out_fc

    def pad_and_merge(self, x0, x1, x2, x3, idx_tensor):
        x = torch.cat((x0,x1,x2,x3), 3)
        return  x

class SoftArgmax1D(torch.nn.Module):
    """
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """
    def __init__(self, device='cuda:0', base_index=0, step_size=1):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....
        Assumes that the input to this layer will be a batch of 1D tensors (so a 2D tensor).
        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        """
        super(SoftArgmax1D, self).__init__()
        self.device = device
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax(x) = \sum_i (i * softmax(x)_i)
        :param x: The input to the soft arg-max layer
        :return: Output of the soft arg-max layer
        """
        smax = self.softmax(x)
        end_index = self.base_index + x.size()[1] * self.step_size
        indices = torch.arange(start=self.base_index, end=end_index, step=self.step_size)
        indices = indices.to(self.device)
        return torch.matmul(smax, indices.type(torch.cuda.FloatTensor))

"""
 Weight Initialization
 ======================
"""
def init_weights(net):
    class_name = net.__class__.__name__
    if class_name.find('Conv') != -1:
        net.weight.data.normal_(0.0,0.02)
    elif class_name.find('BatchNorm2d') != -1:
        net.weight.data.normal_(1.0, 0.02)
        net.bias.data.fill_(0)
