import torch
import torch.nn as nn

from utils.unet_func import timestep_embedding

# attention

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1) 

        self.conv1 = nn.Conv2d(in_planes,
                               in_planes // ratio,
                               kernel_size=(1, 1),
                               bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio,
                               in_planes,
                               kernel_size=(1, 1),
                               bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out  # 加和操作
        return self.sigmoid(out)  # sigmoid激活操作

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(2,
                               1,
                               kernel_size,
                               padding=padding,
                               bias=False)
        self.sigmoid = nn.Sigmoid()

        self.hot = None

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        self.hot = self.sigmoid(x)
        return self.hot

class cbamblock(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super().__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x



class attn_ori(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel

        self.norm = torch.nn.GroupNorm(num_groups=32,
                                       num_channels=channel,
                                       eps=1e-6,
                                       affine=True)
        self.q = torch.nn.Conv2d(channel,
                                 channel,
                                 kernel_size=(1, 1),
                                 stride=(1, 1),
                                 padding=0)
        self.k = torch.nn.Conv2d(channel,
                                 channel,
                                 kernel_size=(1, 1),
                                 stride=(1, 1),
                                 padding=0)
        self.v = torch.nn.Conv2d(channel,
                                 channel,
                                 kernel_size=(1, 1),
                                 stride=(1, 1),
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(channel,
                                        channel,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape

        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


# noise

class NoiseInjection(nn.Module):
    def __init__(self, weight):

        super().__init__()

        self.weight = weight

    def forward(self, image):

        batch, channel, height, width = image.shape
        noise = image.new_empty(batch, channel, height, width).normal_()
        return image + self.weight * noise

# down

class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):

        super().__init__()

        block1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=0),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        block2 = nn.Sequential(
            # nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1, stride=2),
            nn.MaxPool2d(2),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        skip = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1)),
        )

        self.block1 = block1
        self.block2 = block2
        self.skip = skip

    def forward(self, x):
        x = self.block1(x) + self.skip(x)
        res = self.block2(x)
        return res

# up

class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):

        super().__init__()

        block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        skip = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(1, 1)),
        )

        self.block1 = block1
        self.block2 = block2
        self.skip = skip


    def forward(self, x):
        x = self.block1(x) + self.skip(x)
        res = self.block2(x)
        return res


# res

class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0.0,
                 temb_channels=512):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=32,
                         num_channels=in_channels,
                         eps=1e-6,
                         affine=True),
            nn.SiLU(),
            torch.nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
        )

        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=32,
                         num_channels=out_channels,
                         eps=1e-6,
                         affine=True),
            nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(out_channels,
                            out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
        )

        self.deconv = torch.nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      padding=0)

        self.silu = nn.SiLU()

    def forward(self, input, temb):
        # `temb` has shape `[batch_size, time_dim]`

        x = self.block1(input)
        time = self.temb_proj(self.silu(temb))[:, :, None, None]
        # if tensor t shape is [2,3], t[:,None,:,None] shape is [2,1,3,1]
        x = x + time

        x = self.block2(x)

        if self.in_channels != self.out_channels:
            input = self.deconv(input)

        return x + input


class TimeEmbeddingBlock(nn.Module):
    def __init__(self, in_channel, out_channel):

        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.line1 = torch.nn.Linear(self.in_channel, self.out_channel)
        self.silu = nn.SiLU()
        self.line2 = torch.nn.Linear(self.out_channel, self.out_channel)


    def forward(self, x):
        res = timestep_embedding(x, self.in_channel)
        res = self.line1(res)
        res = self.silu(res)
        res = self.line2(res)

        return res
