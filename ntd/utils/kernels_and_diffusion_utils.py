import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from scipy.linalg import cholesky_banded, solve_banded

### Kernels


class OUProcess(nn.Module):
    """
    Ornstein-Uhlenbeck process.
    Provides a sample method and a Mahalabonis distance method.
    Supports linear time operations.

    Args:
        sigma_squared: Variance of the process.
        ell: Length scale of the process.
        signal_length: Length of the signal to sample and compute the distance on.
    """

    def __init__(self, sigma_squared, ell, signal_length):
        super().__init__()
        self.sigma_squared = sigma_squared
        self.ell = ell
        self.signal_length = signal_length
        self.device = "cpu"  # default

        # Build banded precision (only diag and lower diag) because of symmetry.
        lower_banded = np.zeros((2, signal_length))
        lower_banded[0, 1:-1] = _mid_diag(ell, sigma_squared)
        lower_banded[0, 0] = _corner_diag(ell, sigma_squared)
        lower_banded[0, -1] = _corner_diag(ell, sigma_squared)
        lower_banded[1, :-1] = _off_diag(ell, sigma_squared)

        banded_lower_prec_numpy = cholesky_banded(lower_banded, lower=True)
        # Transpose as needed, matrix now in upper notation as a result.
        self.banded_upper_prec_numpy = np.zeros((2, signal_length))
        self.banded_upper_prec_numpy[0, 1:] = banded_lower_prec_numpy[1, :-1]
        self.banded_upper_prec_numpy[1, :] = banded_lower_prec_numpy[0, :]

        # Convert to torch tensor
        self.register_buffer(
            "banded_upper_prec",
            torch.from_numpy(np.float32(self.banded_upper_prec_numpy)),
        )

        self.register_buffer(
            "dense_upper_matrix",
            torch.diag(self.banded_upper_prec[0, 1:], diagonal=1)
            + torch.diag(self.banded_upper_prec[1, :], diagonal=0),
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]
        return self

    def sqrt_mal(self, train_batch):  # (B, C, L)
        assert self.signal_length == train_batch.shape[2]
        upper_diag = self.banded_upper_prec[0]
        main_diag = self.banded_upper_prec[1]
        upper_mult = torch.einsum("l, bcl -> bcl", upper_diag, train_batch)
        main_mult = torch.einsum("l, bcl -> bcl", main_diag, train_batch)
        main_mult[:, :, :-1] += upper_mult[:, :, 1:]
        return main_mult

    def sample(self, sample_shape):
        return self.sample_gpu_dense(sample_shape)

    # O(n)
    def sample_numpy_banded(self, sample_shape):
        normal_samples = np.random.randn(*sample_shape)
        ou_samples = solve_banded(
            (0, 1),  # Upper triangular matrix.
            self.banded_upper_prec_numpy,
            np.transpose(normal_samples, (2, 1, 0)),
        )
        return torch.from_numpy(np.float32(np.transpose(ou_samples, (2, 1, 0)))).to(
            self.device
        )

    # This is not O(n), but for shorter sequences, the theoretical advantage is dwarfed by GPU acceleration.
    def sample_gpu_dense(self, sample_shape):
        normal_samples = torch.randn(*sample_shape, device=self.device)
        res = torch.linalg.solve_triangular(
            self.dense_upper_matrix, torch.transpose(normal_samples, 1, 2), upper=True
        )
        return torch.transpose(res, 1, 2)


def _off_diag(ell, sigma_squared):
    """Helper function of banded OU precision matrix."""
    return (1.0 / sigma_squared) * (np.exp(-(1 / ell))) / (np.exp(-(2 / ell)) - 1.0)


def _corner_diag(ell, sigma_squared):
    """Helper function of banded OU precision matrix."""
    return (1.0 / sigma_squared) * (1.0 / (1.0 - np.exp(-(2 / ell))))


def _mid_diag(ell, sigma_squared):
    """Helper function of banded OU precision matrix."""
    return (1.0 / sigma_squared) * (
        (1.0 + np.exp(-(2 / ell))) / (1.0 - np.exp(-(2 / ell)))
    )


class WhiteNoiseProcess(nn.Module):
    """
    White noise process.
    Provides a sample method and a Mahalabonis distance method.
    In the case of white noise, this is just the (scaled) L2 distance.

    Args:
        sigma_squared: Variance of the white noise.
        signal_length: Length of the signal to sample and compute the distance on.
    """

    def __init__(self, sigma_squared, signal_length):
        super().__init__()
        self.sigma_squared = sigma_squared
        self.signal_length = signal_length  # needs to be implemented
        self.device = "cpu"

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]
        return self

    # Expects and returns tensor with shape (B, C, L).
    def sample(self, sample_shape):
        return np.sqrt(self.sigma_squared) * torch.randn(
            *sample_shape, device=self.device
        )

    def sqrt_mal(self, train_batch):
        return (1 / self.sigma_squared) * train_batch


def generate_samples(diffusion, total_num_samples, batch_size, cond=None):
    """
    Generate samples from a diffusion model.

    Args:
        diffusion: Trained diffusion model.
        total_num_samples: Total number of samples to generate.
        batch_size: Batch size to use for sampling.
        cond: Conditioning information. Can either be the same shape as signal_tensor or a single
            condition to be repeated.

    Returns:
        Generated samples.
    """

    batch_size_ls = [batch_size] * (total_num_samples // batch_size)
    if (remainder := total_num_samples % batch_size) > 0:
        batch_size_ls += [remainder]

    if cond is not None:
        if len(cond.shape) == 2:
            cond = repeat(cond, "c t -> b c t", b=total_num_samples)
        elif len(cond.shape) == 3:
            assert cond.shape[0] == total_num_samples
        else:
            raise ValueError("Cond must be a 2D or 3D tensor!")

    samples_ls = []
    running_batch_count = 0
    for bs in batch_size_ls:
        if cond is None:
            batch_cond = cond
        else:
            batch_cond = cond[running_batch_count : running_batch_count + bs]
            batch_cond = batch_cond.to(diffusion.device)

        samples = diffusion.sample(
            bs,
            cond=batch_cond,
        )
        samples = samples.cpu()
        samples_ls.append(samples)
        running_batch_count += bs

    return torch.cat(samples_ls, dim=0)


def impute_samples(diffusion, samples, mask, batch_size, cond=None):
    """
    Impute missing values in samples using a diffusion model.

    Args:
        diffusion: Trained diffusion model.
        samples: Tensor of samples with missing values.
        mask: Binary task tensor indicating missing values (0 for missing, 1 for observed).
        batch_size: Batch size to use for imputation.
        cond: Conditioning information. Can either be the same shape as samples or a single
            condition to be repeated.

    Returns:
        Imputed samples.
    """

    batch_size_ls = [batch_size] * (samples.shape[0] // batch_size)
    if (remainder := samples.shape[0] % batch_size) > 0:
        batch_size_ls += [remainder]
    print(batch_size_ls)

    if len(mask.shape) == 2:
        mask = repeat(mask, "c t -> b c t", b=samples.shape[0])
    elif len(mask.shape) == 3:
        assert mask.shape[0] == samples.shape[0]
    else:
        raise ValueError("Mask must be a 2D or 3D tensor!")

    if cond is not None:
        if len(cond.shape) == 2:
            cond = repeat(cond, "c t -> b c t", b=samples.shape[0])
        elif len(cond.shape) == 3:
            assert cond.shape[0] == samples.shape[0]
        else:
            raise ValueError("Cond must be a 2D or 3D tensor!")

    imputed_samples_ls = []
    running_batch_count = 0
    for bs in batch_size_ls:
        if cond is None:
            batch_cond = cond
        else:
            batch_cond = cond[running_batch_count : running_batch_count + bs]
            batch_cond = batch_cond.to(diffusion.device)

        imputed_samples = diffusion.impute(
            samples[running_batch_count : running_batch_count + bs].to(
                diffusion.device
            ),
            mask[running_batch_count : running_batch_count + bs].to(diffusion.device),
            cond=batch_cond,
        )
        imputed_samples = imputed_samples.cpu()
        imputed_samples_ls.append(imputed_samples)
        running_batch_count += bs

    return torch.cat(imputed_samples_ls, dim=0)


def impute_single_sample(
    diffusion, sample, mask, num_imputations, batch_size=None, cond=None
):
    """
    Impute missing values in a single sample using a diffusion model.

    Args:
        diffusion: Trained diffusion model.
        sample: Single sample with missing values.
        mask: Binary task tensor indicating missing values (0 for missing, 1 for observed).
        num_imputations: Number of imputations to perform for the single sample.
        batch_size: Batch size to use for imputation.
        cond: Conditioning information. Can either be the same shape as sample or a single
            condition to be repeated.

    Returns:
        Imputed samples.
    """

    assert len(sample.shape) == 2

    sample = repeat(sample, "c t -> b c t", b=num_imputations)

    if batch_size is None:
        batch_size = num_imputations
    return impute_samples(diffusion, sample, mask, batch_size, cond)


def impute_dataset(diffusion, dataset, mask, batch_size=100):
    """
    Impute missing values for each sample in a given dataset using a diffusion model.

    Args:
        diffusion: Trained diffusion model.
        dataset: Dataset of samples with missing values.
        mask: Binary task tensor indicating missing values (0 for missing, 1 for observed).
              Same mask will be used for all samples in the dataset.
        batch_size: Batch size to use for imputation.

    Returns:
        Imputed samples.
    """

    samples = torch.stack([dic["signal"] for dic in dataset])
    try:
        cond = torch.stack([dic["cond"] for dic in dataset])
    except KeyError:
        cond = None
    return impute_samples(diffusion, samples, mask, batch_size, cond)


def impute_dataset_channelwise(diffusion, dataset, channel=[0], batch_size=100):
    """
    Impute whole channels for each sample in a given dataset using a diffusion model.

    Args:
        diffusion: Trained diffusion model.
        test_dataset: Dataset of samples with missing channels.
        channel: List of channels to impute.
        batch_size: Batch size to use for imputation.

    Returns:
        Imputed samples.
    """

    example_sample = dataset[0]["signal"]  # get the shape
    channel_mask = torch.ones_like(example_sample)
    channel_mask[channel] = 0.0
    return impute_dataset(diffusion, dataset, channel_mask, batch_size)
