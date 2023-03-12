import torch
import torch.nn as nn

from part import cbamblock, attn_ori
from part import NoiseInjection
from part import UpSample, DownSample
from part import ResnetBlock

from part import TimeEmbeddingBlock


class DDPM(nn.Module):
    def __init__(self,
                 channel=128,
                 ch_mult=(1, 1, 2, 2, 4, 4),
                 num_res_blocks=2,
                 attn_resolutions=(16, 32),
                 dropout=0.0,
                 resolution=256,
                 ):

        super().__init__()

        self.channel = channel
        self.temb_channel = self.channel * 4
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        # timestep embedding
        self.timeblock = TimeEmbeddingBlock(self.channel, self.temb_channel)

        # downsampling
        self.preprocess = torch.nn.Conv2d(3,
                                          self.channel,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=1)

        curr_res = resolution

        self.down = nn.ModuleList()

        for i_level in range(len(self.ch_mult) - 1):

            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = self.channel * self.ch_mult[i_level]
            block_out = self.channel * self.ch_mult[i_level + 1]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_channel,
                                         dropout=dropout))

                if curr_res in self.attn_resolutions:
                    attn.append(attn_ori(block_out))

                block_in = block_out

            downsample = DownSample(block_out, block_out)

            down = nn.Module()
            down.block = block
            down.attn = attn
            down.downsample = downsample

            curr_res = curr_res // 2

            self.down.append(down)

        # middle
        self.resblock1 = ResnetBlock(in_channels=self.channel * self.ch_mult[-1],
                                     out_channels=self.channel * self.ch_mult[-1],
                                     temb_channels=self.temb_channel,
                                     dropout=dropout)
        self.resattn = attn_ori(self.channel * self.ch_mult[-1])
        self.resblock2 = ResnetBlock(in_channels=self.channel * self.ch_mult[-1],
                                     out_channels=self.channel * self.ch_mult[-1],
                                     temb_channels=self.temb_channel,
                                     dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in range(len(self.ch_mult) - 1, 0, -1):

            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = self.channel * self.ch_mult[i_level] * 2
            block_out = self.channel * self.ch_mult[i_level - 1]

            for i_block in range(self.num_res_blocks):

                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_channel,
                                         dropout=dropout))

                if curr_res in self.attn_resolutions:
                    attn.append(attn_ori(block_out))

                block_in = block_out

            upsample = UpSample(block_out, block_out)

            up = nn.Module()
            up.block = block
            up.attn = attn
            up.upsample = upsample

            curr_res = curr_res * 2

            self.up.insert(0, up)

        # end
        self.postprocess = nn.Sequential(
            nn.GroupNorm(num_groups=32,
                         num_channels=self.channel,
                         eps=1e-6,
                         affine=True),
            nn.SiLU(),
            torch.nn.Conv2d(self.channel,
                            3,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
        )

    def forward(self, x, t):

        # timestep embedding
        temb = self.timeblock(t)

        # downsampling
        left_li = []

        x = self.preprocess(x)
        left_li.append(x)

        for i_level in range(len(self.ch_mult) - 1):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x, temb)
                if len(self.down[i_level].attn) > 0:
                    x = self.down[i_level].attn[i_block](x)
            x = self.down[i_level].downsample(x)
            left_li.append(x)

        # middle
        x = self.resblock1(x, temb)
        x = self.resattn(x)
        x = self.resblock2(x, temb)

        # upsampling
        for i_level in range(len(self.ch_mult) - 1, 0, -1):
            x = torch.cat([x, left_li[i_level]], dim=1)
            for i_block in range(self.num_res_blocks):
                x = self.up[i_level - 1].block[i_block](x, temb)
                if len(self.up[i_level - 1].attn) > 0:
                    x = self.up[i_level - 1].attn[i_block](x)
            x = self.up[i_level - 1].upsample(x)

        # end
        x = self.postprocess(x)

        return x
