import math

import torch
import torch.nn as nn
from einops import rearrange, repeat

from ntd.modules import (
    EfficientMaskedConv1d,
    SLConv,
    get_in_mask,
    get_mid_mask,
    get_out_mask,
)


class CatConvBlock(nn.Module):

    def __init__(
        self,
        hidden_channel_full,
        slconv_kernel_size,
        num_scales,
        heads,
        use_fft_conv,
        padding_mode,
        mid_mask,
    ):
        super().__init__()
        self.block = nn.Sequential(
            SLConv(
                num_channels=hidden_channel_full,
                kernel_size=slconv_kernel_size,
                num_scales=num_scales,
                heads=heads,
                padding_mode=padding_mode,
                use_fft_conv=use_fft_conv,
            ),
            nn.BatchNorm1d(heads * hidden_channel_full),
            nn.GELU(),
            EfficientMaskedConv1d(
                in_channels=heads * hidden_channel_full,
                out_channels=hidden_channel_full,
                kernel_size=1,
                mask=mid_mask,
                bias=False,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(hidden_channel_full),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class CatConv(nn.Module):
    """
    Denoising network with structured, long convolutions.

    This implementation uses a fixed number of layers and  was used for all experiments.
    For a more flexible implementation, see SkipLongConv.
    """

    def __init__(
        self,
        signal_length=100,
        signal_channel=1,
        time_dim=10,
        cond_channel=0,
        hidden_channel=20,
        in_kernel_size=17,
        out_kernel_size=17,
        slconv_kernel_size=17,
        num_scales=5,
        heads=1,
        num_blocks=3,
        num_off_diag=20,
        use_fft_conv=False,
        padding_mode="zeros",
        use_pos_emb=False,
    ):
        """
        Args:
            signal_length: Length of the signals used for training.
            signal_channel: Number of signal channels.
            time_dim: Number of diffusion time embedding dimensions.
            cond_channel: Number of conditioning channels.
            hidden_channel: Number of hidden channels per signal channel.
                Total number of hidden channels will be signal_channel * hidden_channel.
            in_kernel_size: Kernel size of the first convolution.
            out_kernel_size: Kernel size of the last convolution.
            slconv_kernel_size: Kernel size used to create the structured long convolutions.
            num_scales: Number of scales used in the structured long convolutions.
            heads: Number of heads used in the structured long convolutions.
            in_mask_mode: Sparsity used for input convolution.
            num_off_diag: Sparsity used for intermediate convolutions.
            out_mask_mode: Sparsity used for output convolution.
            use_fft_conv: Use FFT convolution instead of standard convolution.
            padding_mode: Padding mode. Either "zeros" or "circular".
            activation_type: Activation function used in the network.
            norm_type: Normalization used in the network.
        """

        super().__init__()
        self.signal_length = signal_length  # train signal length
        self.signal_channel = signal_channel
        self.time_dim = time_dim
        self.use_pos_emb = use_pos_emb

        hidden_channel_full = hidden_channel * signal_channel
        cat_time_dim = 2 * time_dim if use_pos_emb else time_dim
        in_channel = signal_channel + cat_time_dim + cond_channel

        in_mask = get_in_mask(
            signal_channel, hidden_channel, cat_time_dim + cond_channel
        )
        mid_mask = (
            None
            if num_off_diag == hidden_channel
            else get_mid_mask(signal_channel, hidden_channel, num_off_diag, heads)
        )
        out_mask = get_out_mask(signal_channel, hidden_channel)

        self.conv_in = nn.Sequential(
            EfficientMaskedConv1d(
                in_channels=in_channel,
                out_channels=hidden_channel_full,
                kernel_size=in_kernel_size,
                mask=in_mask,
                bias=False,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(hidden_channel_full),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                CatConvBlock(
                    hidden_channel_full=hidden_channel_full,
                    slconv_kernel_size=slconv_kernel_size,
                    num_scales=num_scales,
                    heads=heads,
                    use_fft_conv=use_fft_conv,
                    padding_mode=padding_mode,
                    mid_mask=mid_mask,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_out = EfficientMaskedConv1d(
            in_channels=hidden_channel_full,
            out_channels=self.signal_channel,
            kernel_size=out_kernel_size,
            mask=out_mask,
            bias=True,  # last layer
            padding_mode=padding_mode,
        )

    def forward(self, sig, t, cond=None):

        if cond is not None:
            sig = torch.cat([sig, cond], dim=1)

        if self.use_pos_emb:
            pos_emb = TimestepEmbedder.timestep_embedding(
                torch.arange(self.signal_length, device=sig.device),
                self.time_dim,
            )
            pos_emb = repeat(pos_emb, "l c -> b c l", b=sig.shape[0])
            sig = torch.cat([sig, pos_emb], dim=1)

        time_emb = TimestepEmbedder.timestep_embedding(t, self.time_dim)
        time_emb = repeat(time_emb, "b t -> b t l", l=sig.shape[2])
        sig = torch.cat([sig, time_emb], dim=1)

        sig = self.conv_in(sig)
        for block in self.blocks:
            sig = block(sig)
        sig = self.conv_out(sig)
        return sig


class GeneralEmbedder(nn.Module):
    def __init__(self, cond_channel, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_channel, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, cond):
        cond = rearrange(cond, "b c l -> b l c")
        cond = self.mlp(cond)
        return rearrange(cond, "b l c -> b c l")


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class AdaConvBlock(nn.Module):
    def __init__(
        self,
        kernel_size,
        channel,
        num_scales,
        signal_length=1200,
        mid_mask=None,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_scales = num_scales
        self.mid_mask = mid_mask

        self.conv = SLConv(
            self.kernel_size,
            channel,
            num_scales=self.num_scales,
            padding_mode=padding_mode,
            use_fft_conv=use_fft_conv,
        )
        self.mlp = nn.Sequential(
            EfficientMaskedConv1d(channel, channel, 1, mask=self.mid_mask),
            nn.GELU(),
            EfficientMaskedConv1d(channel, channel, 1, mask=self.mid_mask),
        )

        self.norm1 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        self.norm2 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)

        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel // 3, channel * 6, bias=True),
        )

        self.ada_ln[-1].weight.data.zero_()
        self.ada_ln[-1].bias.data.zero_()

    def forward(self, x, t_cond):
        y = x
        y = self.norm1(y)
        temp = self.ada_ln(rearrange(t_cond, "b c l -> b l c"))
        shift_tm, scale_tm, gate_tm, shift_cm, scale_cm, gate_cm = rearrange(
            temp, "b l c -> b c l"
        ).chunk(6, dim=1)
        y = modulate(y, shift_tm, scale_tm)
        y = self.conv(y)
        y = x + gate_tm * y

        x = y
        y = self.norm2(y)
        y = modulate(y, shift_cm, scale_cm)
        y = x + gate_cm * self.mlp(y)
        return y


class AdaConv(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channel,
        cond_dim=0,
        hidden_channel=8,
        in_kernel_size=1,
        out_kernel_size=1,
        slconv_kernel_size=17,
        num_scales=5,
        num_blocks=3,
        num_off_diag=8,
        use_pos_emb=False,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.signal_channel = signal_channel
        self.signal_length = signal_length
        self.use_pos_emb = use_pos_emb

        hidden_channel_full = signal_channel * hidden_channel

        in_mask = get_in_mask(signal_channel, hidden_channel, 0)
        mid_mask = (
            None
            if num_off_diag == hidden_channel
            else get_mid_mask(signal_channel, hidden_channel, num_off_diag, 1)
        )
        out_mask = get_out_mask(signal_channel, hidden_channel)

        self.conv_in = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.blocks = nn.ModuleList(
            [
                AdaConvBlock(
                    slconv_kernel_size,
                    hidden_channel_full,
                    num_scales,
                    signal_length=signal_length,
                    mid_mask=mid_mask,
                    padding_mode=padding_mode,
                    use_fft_conv=use_fft_conv,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_out = EfficientMaskedConv1d(
            hidden_channel_full,
            signal_channel,
            out_kernel_size,
            out_mask,
            padding_mode=padding_mode,
        )

        self.t_emb = TimestepEmbedder(hidden_channel_full // 3)
        if self.use_pos_emb:
            self.pos_emb = TimestepEmbedder(hidden_channel_full // 3)
        if cond_dim > 0:
            self.cond_emb = GeneralEmbedder(cond_dim, hidden_channel_full // 3)

    def forward(self, x, t, cond=None):
        x = self.conv_in(x)

        t_emb = self.t_emb(t.to(x.device))
        t_emb = repeat(t_emb, "b c -> b c l", l=self.signal_length)

        pos_emb = 0
        if self.use_pos_emb:
            pos_emb = self.pos_emb(torch.arange(self.signal_length).to(x.device))
            pos_emb = repeat(pos_emb, "l c -> b c l", b=x.shape[0])

        cond_emb = 0
        if cond is not None:
            cond_emb = self.cond_emb(cond)

        emb = t_emb + pos_emb + cond_emb

        for block in self.blocks:
            x = block(x, emb)

        x = self.conv_out(x)
        return x
