import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

log = logging.getLogger(__name__)


def activation_factory(activation):
    """
    Returns an instance of  the specified activation function.

    Args:
        activation: Name of the activation function.

    Returns:
        Instance of the specified activation function.
    """

    if activation == "identity":
        return nn.Identity()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError


def norm_factory(norm_type, channel):
    """
    Returns an instance of the specified normalization layer.

    Args:
        norm_type: Name of the normalization layer.
        channel: Number of channels of the input.

    Returns:
        Instance of the specified normalization layer.
    """

    if norm_type == "batch_norm":
        return nn.BatchNorm1d(channel)
    else:
        raise NotImplementedError


class SkipConv1d(nn.Module):
    """
    1D Convolutional layer with skip connection.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        self.transposed_linear = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, input):
        return self.conv.forward(input) + self.transposed_linear.forward(input)


class EfficientMaskedConv1d(nn.Module):
    """
    1D Convolutional layer with masking.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mask=None,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()

        if mask is None:
            self.layer = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                bias=bias,
                padding="same",
                padding_mode=padding_mode,
            )
        else:
            self.layer = MaskedConv1d(
                in_channels,
                out_channels,
                kernel_size,
                mask,
                bias=bias,
                padding_mode=padding_mode,
            )

    def forward(self, x):
        return self.layer.forward(x)


class MaskedConv1d(nn.Module):
    """
    1D Convolutional layer with masking.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mask,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        assert (out_channels, in_channels) == mask.shape

        self.padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        total_padding = kernel_size - 1
        left_pad = total_padding // 2
        self.pad = [left_pad, total_padding - left_pad]

        init_k = np.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = nn.Parameter(
            data=torch.FloatTensor(out_channels, in_channels, kernel_size).uniform_(
                -init_k, init_k
            ),
            requires_grad=True,
        )
        self.register_buffer("mask", mask)
        self.bias = (
            nn.Parameter(
                data=torch.FloatTensor(out_channels).uniform_(-init_k, init_k),
                requires_grad=True,
            )
            if bias
            else None
        )

    def forward(self, x):
        return F.conv1d(
            F.pad(x, self.pad, mode=self.padding_mode),
            self.weight * self.mask.unsqueeze(-1),
            self.bias,
        )


class SLConv(nn.Module):
    """
    Structured Long Convolutional layer.
    Adapted from https://github.com/ctlllll/SGConv

    Args:
        kernel_size: Kernel size used to build convolution.
        num_channels: Number of channels.
        num_scales: Number of scales.
            Overall length will be: kernel_size * (2 ** (num_scales - 1))
        decay_min: Minimum decay.
        decay_max: Maximum decay.
        heads: Number of heads.
        padding_mode: Padding mode. Either "zeros" or "circular".
        use_fft_conv: Whether to use FFT convolution.
        interpolate_mode: Interpolation mode. Either "nearest" or "linear".
    """

    def __init__(
        self,
        kernel_size,
        num_channels,
        num_scales,
        decay_min=2.0,
        decay_max=2.0,
        heads=1,
        padding_mode="zeros",
        use_fft_conv=False,
        interpolate_mode="nearest",
    ):
        super().__init__()
        assert decay_min <= decay_max

        self.h = num_channels
        self.num_scales = num_scales
        self.kernel_length = kernel_size * (2 ** (num_scales - 1))

        self.heads = heads

        self.padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.use_fft_conv = use_fft_conv
        self.interpolate_mode = interpolate_mode

        self.D = nn.Parameter(torch.randn(self.heads, self.h))

        total_padding = self.kernel_length - 1
        left_pad = total_padding // 2
        self.pad = [left_pad, total_padding - left_pad]

        # Init of conv kernels. There are more options here.
        # Full kernel is always normalized by initial kernel norm.
        self.kernel_list = nn.ParameterList()
        for _ in range(self.num_scales):
            kernel = nn.Parameter(torch.randn(self.heads, self.h, kernel_size))
            self.kernel_list.append(kernel)

        # Support multiple scales. Only makes sense in non-sparse setting.
        self.register_buffer(
            "multiplier",
            torch.linspace(decay_min, decay_max, self.h).view(1, -1, 1),
        )
        self.register_buffer("kernel_norm", torch.ones(self.heads, self.h, 1))
        self.register_buffer(
            "kernel_norm_initialized", torch.tensor(0, dtype=torch.bool)
        )

    def forward(self, x):
        signal_length = x.size(-1)

        kernel_list = []
        for i in range(self.num_scales):
            kernel = F.interpolate(
                self.kernel_list[i],
                scale_factor=2 ** (max(0, i - 1)),
                mode=self.interpolate_mode,
            ) * self.multiplier ** (self.num_scales - i - 1)
            kernel_list.append(kernel)
        k = torch.cat(kernel_list, dim=-1)

        if not self.kernel_norm_initialized:
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach()
            self.kernel_norm_initialized = torch.tensor(
                1, dtype=torch.bool, device=k.device
            )
            log.debug(f"Kernel norm: {self.kernel_norm.mean()}")
            log.debug(f"Kernel size: {k.size()}")

        assert k.size(-1) < signal_length
        if self.use_fft_conv:
            k = F.pad(k, (0, signal_length - k.size(-1)))

        k = k / self.kernel_norm

        # Convolution
        if self.use_fft_conv:
            if self.padding_mode == "constant":
                factor = 2
            elif self.padding_mode == "circular":
                factor = 1

            k_f = torch.fft.rfft(k, n=factor * signal_length)  # (C H L)
            u_f = torch.fft.rfft(x, n=factor * signal_length)  # (B H L)
            y_f = torch.einsum("bhl,chl->bchl", u_f, k_f)
            slice_start = self.kernel_length // 2
            y = torch.fft.irfft(y_f, n=factor * signal_length)

            if self.padding_mode == "constant":
                y = y[..., slice_start : slice_start + signal_length]  # (B C H L)
            elif self.padding_mode == "circular":
                y = torch.roll(y, -slice_start, dims=-1)
            y = rearrange(y, "b c h l -> b (h c) l")
        else:
            # Pytorch implements convolutions as cross-correlations! flip necessary
            y = F.conv1d(
                F.pad(x, self.pad, mode=self.padding_mode),
                rearrange(k.flip(-1), "c h l -> (h c) 1 l"),
                groups=self.h,
            )

        # Compute D term in state space equation - essentially a skip connection
        y = y + rearrange(
            torch.einsum("bhl,ch->bchl", x, self.D),
            "b c h l -> b (h c) l",
        )

        return y


### Sparse masks


def get_in_mask(
    signal_channel,
    hidden_channel,
    time_channel,
    cond_channel,
    mode,
):
    """
    Returns the input mask for the specified mode.

    Args:
        signal_channel: Number of signal channels.
        hidden_channel: Number of hidden channels.
        time_channel: Number of diffusion time embedding channels.
        cond_channel: Number of conditioning channels.
        mode: Masking mode. Either "full" or "restricted".
            "full" means that all connections are allowed.
            "restricted" masking means that only connections between a given input signal channel
            and its corresponding hidden channel are allowed.

    Returns:
        Input mask as torch tensor.
    """

    if mode == "full":
        np_mask = get_full(
            signal_channel + cond_channel + time_channel,
            signal_channel * hidden_channel,
        )
    elif mode == "restricted":
        np_mask = np.concatenate(
            (
                get_restricted(signal_channel, 1, hidden_channel),
                get_full(cond_channel + time_channel, signal_channel * hidden_channel),
            ),
            axis=1,
        )
    else:
        raise NotImplementedError
    return torch.from_numpy(np.float32(np_mask))


def get_mid_mask(signal_channel, hidden_channel, mode, num_heads=1):
    """
    Returns the hidden mask for the specified mode.

    Args:
        signal_channel: Number of signal channels.
        hidden_channel: Number of hidden channels.
        mode: Masking mode. Either "full" or "restricted" or "off_diag_<int>".
            "full": All connections are allowed.
            "restricted": Only connections between hidden channels associated with a given signal channel are allowed.
            "off_diag_<int>": Like "restricted", but also allows <int> connections between sets of hidden channels.

    """

    if mode == "full":
        np_mask = get_full(
            signal_channel * hidden_channel, signal_channel * hidden_channel
        )
    elif mode == "restricted":
        np_mask = get_restricted(signal_channel, hidden_channel, hidden_channel)
    elif "off_diag_" in mode:
        num_interaction = int(mode[len("off_diag_") :])
        np_mask = np.maximum(
            get_restricted(signal_channel, hidden_channel, hidden_channel),
            get_sub_interaction(signal_channel, hidden_channel, num_interaction),
        )

    return torch.from_numpy(np.float32(np.repeat(np_mask, num_heads, axis=1)))


def get_out_mask(signal_channel, hidden_channel, mode):
    """
    Returns the output mask for the specified mode.

    Args:
        signal_channel: Number of signal channels.
        hidden_channel: Number of hidden channels.
        mode: Masking mode. Either "full" or "restricted".
            "full": All connections are allowed.
            "restricted": Only connections between a given hidden channel
            and its corresponding output signal channel are allowed.

    Returns:
        Output mask as torch tensor.
    """

    if mode == "full":
        np_mask = get_full(signal_channel * hidden_channel, signal_channel)
    elif mode == "restricted":
        np_mask = get_restricted(signal_channel, hidden_channel, 1)
    return torch.from_numpy(np.float32(np_mask))


def get_full(num_in, num_out):
    """Get full mask containing all ones."""
    return np.ones((num_out, num_in))


def get_restricted(num_signal, num_in, num_out):
    """Get mask with ones only on the block diagonal."""
    return np.repeat(np.repeat(np.eye(num_signal), num_out, axis=0), num_in, axis=1)


def get_sub_interaction(num_signal, size_hidden, num_sub_interaction):
    """Get off-diagonal interactions"""
    sub_interaction = np.zeros((size_hidden, size_hidden))
    sub_interaction[:num_sub_interaction, :num_sub_interaction] = 1.0
    return np.tile(sub_interaction, (num_signal, num_signal))
