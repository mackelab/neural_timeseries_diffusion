"""Adapted from https://github.com/yang-song/score_sde_pytorch"""
import logging

import numpy as np
import torch
from einops import repeat
from scipy import integrate

import ntd.utils.sde_lib as sde_lib
from ntd.utils.utils import standardize_array


log = logging.getLogger(__name__)


def get_model_fn(model, train=False):
    """
    Create a function to give the output of the score-based model.

    Args:
      model: The score model.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, t, cond=None):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.
          Now replaced with t.

        Returns:
          A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(x, t, cond=cond)
        else:
            model.train()
            return model(x, t, cond=cond)

    return model_fn


def get_score_fn(sde, model, train=False):
    """
    Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A score model.
      train: `True` for training and `False` for evaluation.
      REMOVED continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
      A score function.
    """
    model_fn = get_model_fn(model, train=train)

    # VARIANCE PRESERVING DDPM
    def score_fn(x, t, cond=None):
        labels = t * (sde.N - 1)
        score = model_fn(x, labels, cond=cond)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

        score = -score / std[:, None, None]
        return score

    return score_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps, cond=None):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t, cond=cond) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


def get_likelihood_fn(
    sde,
    hutchinson_type="Rademacher",
    rtol=1e-5,
    atol=1e-5,
    method="RK45",
    eps=1e-5,
):
    """
    Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      inverse_scaler: The inverse data normalizer.
      hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
      rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
      atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
      method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.
      eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

    Returns:
      A function that a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """

    def drift_fn(model, x, t, cond=None):
        """The drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False)
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, cond=cond)[0]

    def div_fn(model, x, t, noise, cond=None):
        return get_div_fn(lambda xx, tt, cond: drift_fn(model, xx, tt, cond=cond))(
            x, t, noise, cond=cond
        )

    def likelihood_fn(model, data, cond=None):
        """
        Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
          model: A score model.
          data: A PyTorch tensor.

        Returns:
          A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
          A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
            probability flow ODE.
          An integer. The number of function evaluations used for running the black-box ODE solver.
        """
        if cond is not None:
            assert data.shape[0] == cond.shape[0]

        with torch.no_grad():
            shape = data.shape
            if hutchinson_type == "Gaussian":
                epsilon = torch.randn_like(data)
            elif hutchinson_type == "Rademacher":
                epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.0
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

            def ode_func(t, x):
                sample = (
                    from_flattened_numpy(x[: -shape[0]], shape)
                    .to(data.device)
                    .type(torch.float32)
                )
                log.info(t)
                vec_t = torch.ones(shape[0], device=data.device) * t
                drift = to_flattened_numpy(drift_fn(model, sample, vec_t, cond=cond))
                logp_grad = to_flattened_numpy(
                    div_fn(model, sample, vec_t, epsilon, cond=cond)
                )
                return np.concatenate([drift, logp_grad], axis=0)

            init = np.concatenate(
                [to_flattened_numpy(data), np.zeros((shape[0],))], axis=0
            )
            solution = integrate.solve_ivp(
                ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method
            )
            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = (
                from_flattened_numpy(zp[: -shape[0]], shape)
                .to(data.device)
                .type(torch.float32)
            )
            delta_logp = (
                from_flattened_numpy(zp[-shape[0] :], (shape[0],))
                .to(data.device)
                .type(torch.float32)
            )
            prior_logp = sde.prior_logp(z)
            logp = prior_logp + delta_logp
            return logp, z, nfe

    return likelihood_fn


def compute_likelihoods_batchwise(diffusion, signal_tensor, batch_size, cond=None):
    """
    Compute the likelihoods for a tensor of signals.

    Args:
        diffusion: Trained diffusion model.
        signal_tensor: Tensor of signals.
        batch_size: Batch size used to split up the signal tensor.
        cond: Conditioning information. Can either be the same shape as signal_tensor or a single
            condition to be repeated.

    Returns:
        Tensor of log-likelihoods.
    """

    torch.cuda.empty_cache()

    num_signals, _channel, _signal_length = signal_tensor.shape

    batch_size_ls = [batch_size] * (num_signals // batch_size)
    if (remainder := num_signals % batch_size) > 0:
        batch_size_ls += [remainder]

    if cond is not None and len(cond.shape) == 2:
        cond = repeat(cond, "c t -> b c t", b=num_signals)

    assert diffusion.schedule == "linear"
    sde = sde_lib.VPSDE(
        beta_min=diffusion.start_beta * diffusion.diffusion_time_steps,
        beta_max=diffusion.end_beta * diffusion.diffusion_time_steps,
        N=diffusion.diffusion_time_steps,
    )

    likelihood_computer = get_likelihood_fn(sde=sde)

    ll_ls = []
    running_batch_count = 0
    for bs in batch_size_ls:
        if cond is None:
            batch_cond = cond
        else:
            batch_cond = cond[running_batch_count : running_batch_count + bs]
            batch_cond = batch_cond.to(diffusion.device)

        ll, _z, _nfe = likelihood_computer(
            diffusion.network,
            signal_tensor[running_batch_count : running_batch_count + bs],
            cond=batch_cond,
        )
        ll = ll.cpu()
        ll_ls.append(ll)
        torch.cuda.empty_cache()
        running_batch_count += bs

    all_samples = torch.cat(ll_ls, dim=0)
    return all_samples


### Experiments


def likelihood_experiment(diffusion, test_data, batch_size=100):
    """
    Run the likelihood experiment and the outlier experiment on the test data.

    Args:
        diffusion: Trained diffusion model.
        test_data: Test data.
        batch_size: Batch size for likelihood computation.

    Returns:
        Dictionary with the log-likelihoods under target and control for full and corrupted data.
    """

    real_test_data = torch.stack([dic["signal"] for dic in test_data])
    real_test_data = real_test_data.to(diffusion.device)
    num_test_signals, num_channels, signal_length = real_test_data.shape

    assert num_channels % 2 == 0

    cond_zero = torch.zeros(1, signal_length, device=diffusion.device)
    cond_one = torch.ones(1, signal_length, device=diffusion.device)

    one_ll = compute_likelihoods_batchwise(
        diffusion, real_test_data, batch_size, cond=cond_one
    )
    zero_ll = compute_likelihoods_batchwise(
        diffusion, real_test_data, batch_size, cond=cond_zero
    )

    # broken channels: every other one
    bc = [i % 2 == 0 for i in range(num_channels)]

    # clone real signals
    test_in_broken = torch.clone(real_test_data)
    # and break them by flipping
    flip_axis = 2  # flip timewise
    test_in_broken[:, bc, :] = test_in_broken[:, bc, :].flip(flip_axis)
    one_ll_flip = compute_likelihoods_batchwise(
        diffusion, test_in_broken, batch_size, cond=cond_one
    )
    zero_ll_flip = compute_likelihoods_batchwise(
        diffusion, test_in_broken, batch_size, cond=cond_zero
    )

    # overwrite first channel with white noise
    test_in_broken = torch.clone(real_test_data)
    test_in_broken[:, bc, :] = torch.randn(
        num_test_signals, len(bc) // 2, signal_length, device=diffusion.device
    )
    one_ll_noise = compute_likelihoods_batchwise(
        diffusion, test_in_broken, batch_size, cond=cond_one
    )
    zero_ll_noise = compute_likelihoods_batchwise(
        diffusion, test_in_broken, batch_size, cond=cond_zero
    )

    return {
        "one_ll": one_ll,
        "zero_ll": zero_ll,
        "one_ll_flip": one_ll_flip,
        "zero_ll_flip": zero_ll_flip,
        "one_ll_noise": one_ll_noise,
        "zero_ll_noise": zero_ll_noise,
    }


def long_likelihood_experiment(diffusion, train_dataset, window_arr, batch_size):
    """
    Evaluate the likelihood on a long stretch of  recording.

    Args:
        diffusion: Trained diffusion model.
        train_dataset: Training dataset used for correct standardization.
        window_arr: Array of shape (num_channels, signal_length) containing the signal.
        batch_size: Batch size for likelihood computation.

    Returns:
        Dictionary with the log-likelihoods under target and control.
    """

    stand_arr = standardize_array(
        window_arr,
        ax=None,  # prespecified mean and std
        set_mean=train_dataset.dataset.arr_mean,
        set_std=train_dataset.dataset.arr_std,
    )

    stand_arr_tens = torch.from_numpy(stand_arr).to(diffusion.device)
    likelihoods_ones = compute_likelihoods_batchwise(
        diffusion, stand_arr_tens, batch_size, torch.ones(1, 1000).to(diffusion.device)
    )
    likelihoods_zeros = compute_likelihoods_batchwise(
        diffusion, stand_arr_tens, batch_size, torch.zeros(1, 1000).to(diffusion.device)
    )

    likelihoods_zeros_numpy = likelihoods_zeros.cpu().numpy()
    likelihoods_ones_numpy = likelihoods_ones.cpu().numpy()

    return {"one_ll": likelihoods_ones_numpy, "zero_ll": likelihoods_zeros_numpy}


if __name__ == "__main__":
    pass
