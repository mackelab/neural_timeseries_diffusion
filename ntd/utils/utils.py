import os
import pickle
import tempfile

import numpy as np
import torch
import wandb
from einops import rearrange
from omegaconf import OmegaConf as OC
from scipy import signal

### Util functions


def standardize_array(arr, ax, set_mean=None, set_std=None, return_mean_std=False):
    """
    Standardize array along given axis. set_mean and set_std can be used to manually set mean and standard deviation.

    Args:
        arr: Array to be standardized.
        ax: Axis along which to standardize.
        set_mean: If not None, use this value as mean.
        set_std: If not None, use this value as standard deviation.
        return_mean_std: If True, return mean and standard deviation that were used for standardization.

    Returns:
        Standardized array.
        If return_mean_std is True: Mean
        If return_mean_std is True: Standard deviation
    """

    if set_mean is None:
        arr_mean = np.mean(arr, axis=ax, keepdims=True)
    else:
        arr_mean = set_mean
    if set_std is None:
        arr_std = np.std(arr, axis=ax, keepdims=True)
    else:
        arr_std = set_std

    assert np.min(arr_std) > 0.0
    if return_mean_std:
        return (arr - arr_mean) / arr_std, arr_mean, arr_std
    else:
        return (arr - arr_mean) / arr_std


def surrogates(x, ns, tol_pc=5.0, maxiter=1e6, sorttype="quicksort"):
    """
    Taken from https://github.com/mlcs/iaaft.
    Returns iAAFT surrogates of given time series.

    Args:
        x: Input time series for which IAAFT surrogates are to be estimated.
        ns: Number of surrogates to be generated.
        tol_pc: Tolerance (in percent) level which decides the extent to which the
            difference in the power spectrum of the surrogates to the original
            power spectrum is allowed (default = 5).
        maxiter: Maximum number of iterations before which the algorithm should
            converge. If the algorithm does not converge until this iteration
            number is reached, the while loop breaks.
        sorttype: Type of sorting algorithm to be used when the amplitudes of the newly
            generated surrogate are to be adjusted to the original data. This
            argument is passed on to `numpy.argsort`. Options include: 'quicksort',
            'mergesort', 'heapsort', 'stable'. See `numpy.argsort` for further
            information. Note that although quick sort can be a bit faster than
            merge sort or heap sort, it can, depending on the data, have worse case
            spends that are much slower.

    Returns:
        Array containing the IAAFT surrogates of `x` such that each row of `xs`
        is an individual surrogate time series.
    """

    nx = x.shape[0]
    xs = np.zeros((ns, nx))
    maxiter = 10000
    ii = np.arange(nx)

    x_amp = np.abs(np.fft.fft(x))
    x_srt = np.sort(x)
    r_orig = np.argsort(x)

    for k in range(ns):
        count = 0
        r_prev = np.random.permutation(ii)
        r_curr = r_orig
        z_n = x[r_prev]
        percent_unequal = 100.0

        while (percent_unequal > tol_pc) and (count < maxiter):
            r_prev = r_curr

            y_prev = z_n
            fft_prev = np.fft.fft(y_prev)
            phi_prev = np.angle(fft_prev)
            e_i_phi = np.exp(phi_prev * 1j)
            z_n = np.fft.ifft(x_amp * e_i_phi)

            r_curr = np.argsort(z_n, kind=sorttype)
            z_n[r_curr] = x_srt.copy()
            percent_unequal = (np.sum(r_curr != r_prev) * 100.0) / nx
            count += 1

        if count >= (maxiter - 1):
            print("maximum number of iterations reached!")

        xs[k] = np.real(z_n)

    return xs


def surrogate_dataset(array):
    """
    Apply surrogate method to each signal in array.

    Args:
        array: Array of signals.

    Returns:
        Array of surrogate signals.
    """
    surro_array = np.zeros_like(array)
    for idx, sig in enumerate(array):
        surro_array[idx] = surrogates(sig, 1)[0]
    return surro_array


### Sharp Wave Ripples (SWR) utils


def phase_amplitude_coupling(
    signals_high,
    signals_low,
    low_wn,
    high_wn,
    num_bins=31,
    fs=600,
    filter_order=100,
):
    """
    Computes the phase amplitude coupling (PAC) between two signals.

    Args:
        signals_high: High frequency signal used for amplitude.
        signals_low: Low frequency signal used for phase.
        high_wn: High frequency band.
        low_wn: Low frequency band.
        num_bins: Number of phase bins.
        fs: Sampling frequency.
        filter_order: Filter order for the FIR filters.

    Returns:
        Phase amplitude couplings.
        Phase bins.
    """

    assert signals_high.shape == signals_low.shape
    p_bins = np.linspace(-np.pi, np.pi, num_bins + 1)

    b_low = signal.firwin(
        filter_order, low_wn, fs=fs, pass_zero=False, window="hamming"
    )

    b_high = signal.firwin(
        filter_order, high_wn, fs=fs, pass_zero=False, window="hamming"
    )

    global_pac_sum = np.zeros(num_bins)
    counter = np.zeros(num_bins)
    for idx, (high_sig, low_sig) in enumerate(zip(signals_high, signals_low)):
        Vlo = signal.filtfilt(b_low, 1, low_sig)
        Vhi = signal.filtfilt(b_high, 1, high_sig)

        phi = np.angle(signal.hilbert(Vlo))
        amp = np.abs(signal.hilbert(Vhi))

        for k in range(num_bins):
            pL, pR = p_bins[k], p_bins[k + 1]
            indices = (phi > pL) & (phi <= pR)
            global_pac_sum[k] += np.sum(amp[indices])
            counter[k] += len(amp[indices])

    global_pacs = global_pac_sum / counter
    return global_pacs / np.sum(global_pacs), p_bins


def create_slices(cutter):
    """
    Create slices from a mask array. Slices will correspond to regions of 1s.
    Used to extract SWR positions.

    Args:
        cutter: Mask array of 0s and 1s.

    Returns:
        List of slices.
    """

    temp = np.concatenate([np.zeros(1), cutter, np.zeros(1)])

    diff = temp[1:] - temp[:-1]
    start_pos_arr = np.where(diff == 1)[0]
    end_pos_arr = np.where(diff == -1)[0]

    return [slice(start, end) for start, end in zip(start_pos_arr, end_pos_arr)]


def fuser(mask, gap_fuser):
    """
    Fuse gaps in a mask array. Gaps are repeated 0s.
    If the gap is smaller than gap_fuser, it will be filled with 1s.

    Args:
        mask: Mask array of 0s and 1s.
        gap_fuser: Gap size threshold.

    Returns:
        Mask array with fused gaps.
    """

    gap_slices = create_slices(1 - mask)
    for ss in gap_slices:
        if ss.stop - ss.start <= gap_fuser:
            mask[ss] = 1
    return mask


def filter_short(mask, min_length):
    """
    Filter out short sequences of repeated 1s in a mask array.

    Args:
        mask: Mask array of 0s and 1s.
        min_length: Minimum length of sequence.

    Returns:
        List of slices larger than min_length.
    """

    gap_slices = create_slices(mask)
    return [ss for ss in gap_slices if ss.stop - ss.start >= min_length]


def phase_count_coupling(
    signals_high,
    signals_low,
    low_wn,
    high_wn,
    fuser_gap=10,
    min_length=20,
    num_bins=31,
    filter_order=100,
    fs=600,
    supervised_mean=None,
    supervised_std=None,
):
    """
    Computes the phase count coupling (PCC) between two signals.
    That is, the location of a potential SWR is coupled with the phase of the low frequency signal.

    Args:
        signals_high: High frequency signal used for SWR detection.
        signals_low: Low frequency signal used for phase.
        high_wn: High frequency band.
        low_wn: Low frequency band.
        fuser_gap: Gap size threshold for combining SWRs.
        min_length: Minimum length of SWR.
        num_bins: Number of phase bins.
        filter_order: Filter order for the FIR filters.
        fs: Sampling frequency.
        supervised_mean: If not None, use this value as mean to detect SWRs.
        supervised_std: If not None, use this value as standard deviation to detect SWRs.

    Returns:
        Phase count couplings.
        Phase bins.
    """

    assert signals_high.shape == signals_low.shape

    lst_of_ss_max_pos_ls = extract_sharp_wave_ripples(
        signals_high=signals_high,
        high_wn=high_wn,
        fuser_gap=fuser_gap,
        min_length=min_length,
        supervised_mean=supervised_mean,
        supervised_std=supervised_std,
    )

    b_low = signal.firwin(
        filter_order, low_wn, fs=fs, pass_zero=False, window="hamming"
    )

    p_bins = np.linspace(-np.pi, np.pi, num_bins + 1)

    all_swr_angles = []
    for idx, (ss_max_ls, low_sig) in enumerate(zip(lst_of_ss_max_pos_ls, signals_low)):
        Vlo = signal.filtfilt(b_low, 1, low_sig)
        phi = np.angle(signal.hilbert(Vlo))

        for ss_max_pos in ss_max_ls:
            swr_angle = phi[ss_max_pos]
            all_swr_angles.append(swr_angle)

    bin_ids, counts = np.unique(
        np.digitize(all_swr_angles, p_bins, right=True), return_counts=True
    )

    full_counts = np.zeros(num_bins)
    full_counts[bin_ids - 1] = counts
    return full_counts / np.sum(full_counts), p_bins


def extract_sharp_wave_ripples(
    signals_high,
    high_wn,
    fuser_gap=10,
    min_length=20,
    filter_order=100,
    fs=600,
    std_outlier=3.0,
    supervised_mean=None,
    supervised_std=None,
):
    """
    Extract sharp-wave ripples (SWRs) from a signal array based on outliers in the SWR band.
    If a SWR is detected, the position of the maximum power of filtered signal is returned.

    Args:
        signals_high: Signal used for SWR detection.
        high_wn: High frequency band.
        fuser_gap: Gap size threshold for combining SWRs.
        min_length: Minimum length of SWR.
        filter_order: Filter order for the FIR filters.
        fs: Sampling frequency.
        supervised_mean: If not None, use this value as mean to detect SWRs.
        supervised_std: If not None, use this value as standard deviation to detect SWRs.

    Returns:
        List of lists containing SWR positions.
    """

    filtered_squares, squared_filter_mean, squared_filter_std = filter_squares(
        signals_high=signals_high, high_wn=high_wn, filter_order=filter_order, fs=fs
    )

    if supervised_mean is not None:
        squared_filter_mean = supervised_mean
    if supervised_std is not None:
        squared_filter_std = supervised_std

    lst_of_ss_max_pos_ls = []
    for idx, f_sq in enumerate(filtered_squares):
        mask = np.zeros_like(f_sq)
        indices = f_sq > squared_filter_mean + std_outlier * squared_filter_std
        mask[indices] = 1.0
        swr_slices = filter_short(fuser(mask, fuser_gap), min_length)
        lst_of_ss_max_pos_ls.append(
            [np.argmax(f_sq[ss]) + ss.start for ss in swr_slices]
        )

    return lst_of_ss_max_pos_ls, squared_filter_mean, squared_filter_std


def filter_squares(signals_high, high_wn, filter_order=100, fs=600):
    """
    Filter and square signal array.

    Args:
        signals_high: Signal array.
        high_wn: Frequency band.
        filter_order: Filter order for the FIR filters.
        fs: Sampling frequency.

    Returns:
        Filtered and squared signal array.
        Mean of filtered and squared signal array.
        Standard deviation of filtered and squared signal array.
    """

    b_high = signal.firwin(
        filter_order, high_wn, fs=fs, pass_zero=False, window="hamming"
    )
    filtered_squares = np.zeros_like(signals_high)
    for idx, high_sig in enumerate(signals_high):
        Vhi = signal.filtfilt(b_high, 1, high_sig)
        filtered_squares[idx] = Vhi**2.0

    squared_filter_mean = np.mean(filtered_squares)
    squared_filter_std = np.std(filtered_squares)

    return filtered_squares, squared_filter_mean, squared_filter_std


### Loading and saving


def save_pickle(obj, file_path):
    """
    Save object as pickle file.

    Args:
        obj: Object to be saved.
        file_path: Path to save file to.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        raise FileExistsError(f"{file_path} already exists!")

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def read_pickle(file_name, folder_path):
    """
    Read pickle file.

    Args:
        file_name: Name of file.
        folder_path: Path to folder containing file.
    """
    file_path = os.path.join(folder_path, file_name)
    assert file_path.endswith(".pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def config_saver(
    file,
    filename,
    cfg=None,
    experiment=None,
    tag=None,
    project=None,
    entity=None,
    dir=None,
    mode=None,
    local_path=None,
):
    """
    Save file to wandb or locally.

    Args:
        file: File to be saved.
        filename: Name of file.
        cfg: Config object.
        experiment: Name of wandb experiment.
        tag: Tag for file.
        project: Name of wandb project.
        entity: Name of wandb entity.
        dir: wandb directory.
        mode: wandb mode. "online" for uploading, "disabled" to deactivate wandb.
        local_path: Path to save file to locally.
            None or "". None will check in cfg, "" will not save locally
    """
    # infer params from cfg if not given
    if cfg is not None:
        if experiment is None:
            experiment = cfg.base.experiment
        if tag is None:
            tag = cfg.base.tag
        if project is None:
            project = cfg.base.wandb_project
        if entity is None:
            entity = cfg.base.wandb_entity
        if dir is None:
            dir = cfg.base.home_path
        if mode is None:
            mode = cfg.base.wandb_mode
        if local_path is None:
            local_path = cfg.base.save_path

    # save to wandb
    if mode == "online":
        config = (
            OC.to_container(cfg, resolve=True, throw_on_missing=True)
            if cfg is not None
            else None
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run = wandb.init(
                mode=mode,
                project=project,
                entity=entity,
                dir=dir,
                group=experiment,
                config=config,
                name=f"SAVING_RUN:{tag}_{filename}",
            )
            temp_file_path = os.path.join(tmpdir, filename)
            with open(temp_file_path, "wb") as f:
                pickle.dump(file, f)
            run.save(temp_file_path, base_path=tmpdir)
            run.finish()

    # save locally
    if not (local_path is None or local_path == ""):
        save_pickle(file, os.path.join(local_path, experiment, f"{tag}_{filename}"))


def path_loader(filename, run_path):
    """
    Load file from wandb or locally. First try locally.

    Args:
        filename: Name of file.
        run_path: Local path or wandb run path.

    Returns:
        Loaded file.
    """

    data = None
    try:
        data = read_pickle(filename, run_path)
    except FileNotFoundError:
        with tempfile.TemporaryDirectory() as tmpdir:
            wandb.restore(name=filename, run_path=run_path, root=tmpdir)
            with open(os.path.join(tmpdir, filename), "rb") as f:
                data = pickle.load(f)
            # throws CommError if run not found
            # throws ValueError if file not found
    return data


### Misc


def count_parameters(model):
    """
    Count number of trainable parameters in model.

    Args:
        model: Pytorch model.

    Returns:
        Number of trainable parameters.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def l2_distances(tens_one, tens_two):
    """
    Compute pairwise L2 distances between two signal tensors.

    Args:
        tens_one: First tensor.
        tens_two: Second tensor.

    Returns:
        Pairwise L2 distances.
        Tuple of IDs of two nearest neighbors.
        Nearest neighbor from first signal tensor.
        Nearest neighbor from second signal tensor.
    """

    res = torch.cdist(
        rearrange(tens_one, "b c l -> b (c l)"), rearrange(tens_two, "b c l -> b (c l)")
    )

    flat_argmin = torch.argmin(res).item()
    row = flat_argmin // res.shape[1]
    col = flat_argmin % res.shape[1]

    return res, (row, col), tens_one[row], tens_two[col]


def spectral_error(arr_one, arr_two, fs, nperseg, with_log=True):
    """
    Compute spectral error between two signal arrays.

    Args:
        arr_one: First signal array.
        arr_two: Second signal array.
        fs: Sampling frequency.
        nperseg: Number of samples per segment.
        with_log: If True, compute log of spectral density.

    Returns:
        Spectral error.
    """

    _ff_one, Pxy_one = signal.csd(
        arr_one,
        arr_one,
        axis=1,
        nperseg=nperseg,
        fs=fs,
    )
    _ff_two, Pxy_two = signal.csd(
        arr_two,
        arr_two,
        axis=1,
        nperseg=nperseg,
        fs=fs,
    )
    if with_log:
        Pxy_one = np.log(np.median(Pxy_one, axis=0))
        Pxy_two = np.log(np.median(Pxy_two, axis=0))
    return np.linalg.norm(Pxy_one - Pxy_two)


def total_ll(ll_c1, frac_c1, ll_c2, frac_c2):
    """
    Numerically stable computation of total log likelihood over two classes.
    Uses log(a + b) = log(a) + log(1 + b/a).

    Args:
        ll_c1: Log likelihood of first class.
        frac_c1: Prior class frequency of first class.
        ll_c2: Log likelihood of second class.
        frac_c2: Prior class frequency of second class.

    Returns:
        Total log likelihood.
    """

    def log_trick(ll_a, frac_a, ll_b, frac_b):
        assert ll_a >= ll_b
        log_a = ll_a + np.log(frac_a)
        log_opb = np.log(1.0 + (frac_b / frac_a) * np.exp(ll_b - ll_a))
        return log_a + log_opb

    return (
        log_trick(ll_c1, frac_c1, ll_c2, frac_c2)
        if ll_c1 >= ll_c2
        else log_trick(ll_c2, frac_c2, ll_c1, frac_c1)
    )
