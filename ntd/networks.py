import math

import torch
import torch.nn as nn
from einops import repeat

from ntd.modules import (
    EfficientMaskedConv1d,
    SkipConv1d,
    SLConv,
    activation_factory,
    get_in_mask,
    get_mid_mask,
    get_out_mask,
    norm_factory,
)


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal time embedding.
    """

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.register_buffer("aranged", torch.arange(self.half_dim))

    def forward(self, x):
        emb = math.log(10000.0) / (self.half_dim - 1)
        emb = torch.exp(self.aranged * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class BaseConv(nn.Module):
    def __init__(
        self,
        signal_length=100,
        signal_channel=1,
        time_dim=10,
        cond_channel=0,
        hidden_channel=20,
        out_channel=None,
        kernel_size=29,
        dilation=1,
        norm_type="batch_norm",
        activation_type="leaky_relu",
        padding_mode="zeros",
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.signal_length = signal_length
        self.signal_channel = signal_channel
        self.time_dim = time_dim
        self.cond_channel = cond_channel
        self.in_channel = signal_channel + time_dim + cond_channel
        self.hidden_channel = hidden_channel
        self.out_channel = signal_channel if out_channel is None else out_channel
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (self.dilation * (self.kernel_size - 1)) // 2
        self.padding_mode = padding_mode

        self.time_embbeder = SinusoidalPosEmb(time_dim) if time_dim > 0 else None

        self.conv_pool = nn.Sequential(
            SkipConv1d(
                in_channels=self.in_channel,
                out_channels=self.hidden_channel,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                padding_mode=self.padding_mode,
            ),
            norm_factory(norm_type, self.hidden_channel),
            activation_factory(activation_type),
            SkipConv1d(
                in_channels=self.hidden_channel,
                out_channels=self.hidden_channel,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                padding_mode=self.padding_mode,
            ),
            norm_factory(norm_type, self.hidden_channel),
            activation_factory(activation_type),
            SkipConv1d(
                in_channels=self.hidden_channel,
                out_channels=self.hidden_channel,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                padding_mode=self.padding_mode,
            ),
            norm_factory(norm_type, self.hidden_channel),
            activation_factory(activation_type),
            SkipConv1d(
                in_channels=self.hidden_channel,
                out_channels=self.out_channel,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                padding_mode=self.padding_mode,
            ),
        )

    def forward(self, sig, t, cond=None):  # (-1, channel, len), (-1)
        if cond is not None:
            sig = torch.cat([sig, cond], dim=1)
        if self.time_embbeder is not None:
            time_emb = self.time_embbeder(t)  # (-1, time_dim)
            time_emb_repeat = repeat(time_emb, "b t -> b t l", l=sig.shape[2])
            sig = torch.cat([sig, time_emb_repeat], dim=1)
        return self.conv_pool(sig)


class LongConv(nn.Module):
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
        decay_min=2.0,
        decay_max=2.0,
        heads=1,
        in_mask_mode="full",
        mid_mask_mode="full",
        out_mask_mode="full",
        use_fft_conv=False,
        padding_mode="zeros",
        activation_type="gelu",
        norm_type="batch_norm",
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
            decay_min: Minimum decay of the structured long convolutions.
            decay_max: Maximum decay of the structured long convolutions.
            heads: Number of heads used in the structured long convolutions.
            in_mask_mode: Sparsity used for input convolution.
            mid_mask_mode: Sparsity used for intermediate convolutions.
            out_mask_mode: Sparsity used for output convolution.
            use_fft_conv: Use FFT convolution instead of standard convolution.
            padding_mode: Padding mode. Either "zeros" or "circular".
            activation_type: Activation function used in the network.
            norm_type: Normalization used in the network.
        """

        super().__init__()
        self.signal_length = signal_length  # train signal length
        self.signal_channel = signal_channel
        self.in_channel = signal_channel + time_dim + cond_channel
        self.hidden_channel_full = hidden_channel * signal_channel

        self.time_dim = time_dim
        self.time_embbeder = SinusoidalPosEmb(time_dim) if time_dim > 0 else None

        in_mask = (
            None
            if in_mask_mode == "full"
            else get_in_mask(
                signal_channel, hidden_channel, time_dim, cond_channel, in_mask_mode
            )
        )
        mid_mask = (
            None
            if mid_mask_mode == "full"
            else get_mid_mask(signal_channel, hidden_channel, mid_mask_mode, heads)
        )
        out_mask = (
            None
            if out_mask_mode == "full"
            else get_out_mask(signal_channel, hidden_channel, out_mask_mode)
        )

        self.conv_pool = nn.Sequential(
            EfficientMaskedConv1d(
                in_channels=self.in_channel,
                out_channels=self.hidden_channel_full,
                kernel_size=in_kernel_size,
                mask=in_mask,
                bias=False,
                padding_mode=padding_mode,
            ),
            norm_factory(norm_type, self.hidden_channel_full),
            activation_factory(activation_type),
            SLConv(
                num_channels=self.hidden_channel_full,
                kernel_size=slconv_kernel_size,
                num_scales=num_scales,
                decay_min=decay_min,
                decay_max=decay_max,
                heads=heads,
                padding_mode=padding_mode,
                use_fft_conv=use_fft_conv,
            ),
            norm_factory(norm_type, heads * self.hidden_channel_full),
            activation_factory(activation_type),
            EfficientMaskedConv1d(
                in_channels=heads * self.hidden_channel_full,
                out_channels=self.hidden_channel_full,
                kernel_size=1,
                mask=mid_mask,
                bias=False,
                padding_mode=padding_mode,
            ),
            norm_factory(norm_type, self.hidden_channel_full),
            activation_factory(activation_type),
            SLConv(
                num_channels=self.hidden_channel_full,
                kernel_size=slconv_kernel_size,
                num_scales=num_scales,
                decay_min=decay_min,
                decay_max=decay_max,
                heads=heads,
                padding_mode=padding_mode,
                use_fft_conv=use_fft_conv,
            ),
            norm_factory(norm_type, heads * self.hidden_channel_full),
            activation_factory(activation_type),
            EfficientMaskedConv1d(
                in_channels=heads * self.hidden_channel_full,
                out_channels=self.hidden_channel_full,
                kernel_size=1,
                mask=mid_mask,
                bias=False,
                padding_mode=padding_mode,
            ),
            norm_factory(norm_type, self.hidden_channel_full),
            activation_factory(activation_type),
            SLConv(
                num_channels=self.hidden_channel_full,
                kernel_size=slconv_kernel_size,
                num_scales=num_scales,
                decay_min=decay_min,
                decay_max=decay_max,
                heads=heads,
                padding_mode=padding_mode,
                use_fft_conv=use_fft_conv,
            ),
            norm_factory(norm_type, heads * self.hidden_channel_full),
            activation_factory(activation_type),
            EfficientMaskedConv1d(
                in_channels=heads * self.hidden_channel_full,
                out_channels=self.hidden_channel_full,
                kernel_size=1,
                mask=mid_mask,
                bias=False,
                padding_mode=padding_mode,
            ),
            norm_factory(norm_type, self.hidden_channel_full),
            activation_factory(activation_type),
            EfficientMaskedConv1d(
                in_channels=self.hidden_channel_full,
                out_channels=self.signal_channel,
                kernel_size=out_kernel_size,
                mask=out_mask,
                bias=True,  # in the last layer
                padding_mode=padding_mode,
            ),
        )

    def forward(self, sig, t, cond=None):  # (-1, channel, len), (-1)
        if cond is not None:
            sig = torch.cat([sig, cond], dim=1)
        if self.time_embbeder is not None:
            time_emb = self.time_embbeder(t)  # (-1, time_dim)
            time_emb_repeat = repeat(time_emb, "b t -> b t l", l=sig.shape[2])
            sig = torch.cat([sig, time_emb_repeat], dim=1)
        return self.conv_pool(sig)
