from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve


@dataclass
class FigureLayout:
    """
    Dataclass to define the layout of a figure. That is, width and font size.

    Args:
        width_in_pt: Width of the figure in pt.
        width_grid: Width of the grid in which the figure is placed.
        base_font_size: Base font size of the figure.
        scale_factor: Scale factor of the font size.
            This exposes the factor by which the Figure will be downscaled when included document.
    """

    width_in_pt: float
    width_grid: int
    base_font_size: int = 10
    scale_factor: float = 1.0

    def get_grid_in_inch(self, w_grid, h_grid):
        pt_to_inch = 1 / 72
        assert w_grid <= self.width_grid
        return (
            (w_grid / self.width_grid) * self.width_in_pt * pt_to_inch,
            (h_grid / self.width_grid) * self.width_in_pt * pt_to_inch,
        )

    def get_rc(self, w_grid, h_grid):
        return {
            "figure.figsize": self.get_grid_in_inch(w_grid, h_grid),
            "font.size": self.base_font_size * self.scale_factor,
        }


def basic_plotting(
    fig,
    ax,
    x_label=None,
    x_label_fontsize="medium",
    y_label=None,
    y_label_fontsize="medium",
    x_lim=None,
    y_lim=None,
    x_ticks=None,
    y_ticks=None,
    x_ticklabels=None,
    y_ticklabels=None,
    x_axis_visibility=None,
    y_axis_visibility=None,
):
    """
    Provide some basic plotting functionality.

    Args:
        fig: Figure object.
        ax: Axis object.
        x_label: Label for the x-axis.
        x_label_fontsize: Fontsize of the x-axis label.
        y_label: Label for the y-axis.
        y_label_fontsize: Fontsize of the y-axis label.
        x_lim: Limits of the x-axis.
        y_lim: Limits of the y-axis.
        x_ticks: Ticks of the x-axis.
        y_ticks: Ticks of the y-axis.
        x_ticklabels: Ticklabels of the x-axis.
        y_ticklabels: Ticklabels of the y-axis.
        x_axis_visibility: Visibility of the x-axis.
        y_axis_visibility: Visibility of the y-axis.
    """

    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=x_label_fontsize)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=y_label_fontsize)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    if x_axis_visibility is not None:
        ax.get_xaxis().set_visible(x_axis_visibility)
        try:
            ax.spines["bottom"].set_visible(x_axis_visibility)
        except KeyError:
            print("No bottom spine exists!")
    if y_axis_visibility is not None:
        ax.get_yaxis().set_visible(y_axis_visibility)
        try:
            ax.spines["left"].set_visible(y_axis_visibility)
        except KeyError:
            print("No left spine exists!")

    return fig, ax


def plot_sd(
    fig,
    ax,
    arr_one,
    arr_two,
    fs,
    nperseg,
    agg_function=np.median,
    color_one="red",
    color_two="blue",
    with_quantiles=False,
    alpha=0.1,
    lower_quantile=0.25,
    upper_quantile=0.75,
    alpha_boundary=1.0,
    x_ss=slice(None),
):
    """
    Plot the spectral density of two arrays with pointwise uncertainty.

    Args:
        fig: Figure object.
        ax: Axis object.
        arr_one: First array.
        arr_two: Second array.
        fs: Sampling frequency.
        nperseg: Number of samples per segment.
        agg_function: Aggregation function.
        color_one: Color for the first array.
        color_two: Color for the second array.
        with_quantiles: Whether to plot the quantiles.
        alpha: Alpha value for the quantiles.
        lower_quantile: Lower quantile.
        upper_quantile: Upper quantile.
        alpha_boundary: Alpha value for the percentile boundary.
        x_ss: Frequencies to plot.
    """

    ff_one, Pxy_one = signal.csd(
        arr_one,
        arr_one,
        axis=1,
        nperseg=nperseg,
        fs=fs,
    )
    ff_two, Pxy_two = signal.csd(
        arr_two,
        arr_two,
        axis=1,
        nperseg=nperseg,
        fs=fs,
    )
    if with_quantiles:
        ax.fill_between(
            ff_one[x_ss],
            np.quantile(Pxy_one, lower_quantile, axis=0)[x_ss],
            np.quantile(Pxy_one, upper_quantile, axis=0)[x_ss],
            color=color_one,
            alpha=alpha,
        )
        ax.fill_between(
            ff_two[x_ss],
            np.quantile(Pxy_two, lower_quantile, axis=0)[x_ss],
            np.quantile(Pxy_two, upper_quantile, axis=0)[x_ss],
            color=color_two,
            alpha=alpha,
        )
        ax.loglog(
            ff_one[x_ss],
            np.quantile(Pxy_one, lower_quantile, axis=0)[x_ss],
            color=color_one,
            alpha=alpha_boundary,
        )
        ax.loglog(
            ff_one[x_ss],
            np.quantile(Pxy_one, upper_quantile, axis=0)[x_ss],
            color=color_one,
            alpha=alpha_boundary,
        )
        ax.loglog(
            ff_two[x_ss],
            np.quantile(Pxy_two, lower_quantile, axis=0)[x_ss],
            color=color_two,
            alpha=alpha_boundary,
        )
        ax.loglog(
            ff_two[x_ss],
            np.quantile(Pxy_two, upper_quantile, axis=0)[x_ss],
            color=color_two,
            alpha=alpha_boundary,
        )

    ax.loglog(ff_one[x_ss], agg_function(Pxy_one, axis=0)[x_ss], color=color_one)
    ax.loglog(ff_two[x_ss], agg_function(Pxy_two, axis=0)[x_ss], color=color_two)
    return fig, ax


def plot_phase_line(fig, ax, bin_phase, pac, color="black"):
    """
    Polar plot used for the phase-amplitude coupling.

    Args:
        fig: Figure object.
        ax: Axis object.
        bin_phase: Phase bins.
        pac: Phase-amplitude coupling.
        color: Color of the plotted line.
    """

    ax.plot(
        np.append(bin_phase, bin_phase[0]),
        np.append(pac, pac[0]),
        color=color,
    )
    return fig, ax


def polar_hist(
    fig,
    ax,
    values,
    p_bins,
    grid=10,
    fill_alpha=0.2,
    fillcolor="red",
    spinecolor="black",
    full_spines=False,
):
    """
    Polar histogram used for phase-count coupling.

    Args:
        fig: Figure object.
        ax: Axis object.
        values: Values to plot.
        p_bins: Phase bins.
        grid: Grid size.
        fill_alpha: Alpha value for the fill.
        fillcolor: Color for the fill.
        spinecolor: Color for the spines.
        full_spines: Whether to plot the full spines.
    """
    num_values = len(values)
    assert num_values + 1 == len(p_bins)
    for idx in range(num_values):
        ax.fill_between(
            np.linspace(p_bins[idx], p_bins[idx + 1], grid),
            np.zeros(grid),
            values[idx] * np.ones(grid),
            color=fillcolor,
            alpha=fill_alpha,
        )
        ax.plot(
            np.linspace(p_bins[idx], p_bins[idx + 1], grid),
            values[idx] * np.ones(grid),
            color=spinecolor,
        )
        if full_spines:
            ax.plot(
                [p_bins[idx], p_bins[idx]],
                [0.0, values[idx]],
                color=spinecolor,
            )
            ax.plot(
                [p_bins[idx + 1], p_bins[idx + 1]],
                [0.0, values[idx]],
                color=spinecolor,
            )
        else:
            ax.plot(
                [p_bins[idx], p_bins[idx]],
                [values[idx - 1], values[idx]],
                color=spinecolor,
            )
    return fig, ax


def plot_overlapping_signal(fig, ax, sig, colors=["C0"]):
    """
    Plot signals of a given signal array.

    Args:
        fig: Figure object.
        ax: Axis object.
        sig: Signal array.
        colors: Colors for the individual signal channels.
    """

    if len(colors) == 1:
        colors = len(sig) * colors
    else:
        assert len(colors) == len(sig)

    for chan, col in zip(sig, colors):
        ax.plot(chan, color=col)

    return fig, ax


def plot_density(fig, ax, values, x_range, bw_method=None, d_alpha=0.2, color="C0"):
    """
    Plot the Gaussian kernel density estimate of a given array.

    Args:
        fig: Figure object.
        ax: Axis object.
        values: Values to plot.
        x_range: Values to evalualte the KDE at.
        bw_method: Bandwidth method.
        d_alpha: Alpha value for the fill.
        color: Color of the plot.
    """

    kde = gaussian_kde(values, bw_method=bw_method)
    ax.fill_between(x_range, kde(x_range), alpha=d_alpha, color=color)
    ax.plot(x_range, kde(x_range), color=color)
    return fig, ax


def plot_roc_curve(fig, ax, y_true, y_score, rand_base_col=None):
    """
    Plot the ROC curve.

    Args:
        fig: Figure object.
        ax: Axis object.
        y_true: True labels.
        y_score: Predicted labels.
        rand_base_col: Color for the random baseline. If None, no baseline is plotted.
    """

    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    if rand_base_col is not None:
        ax.plot(
            np.linspace(0.0, 1.0, 100),
            np.linspace(0.0, 1.0, 100),
            color=rand_base_col,
            linestyle="--",
        )
    return fig, ax
