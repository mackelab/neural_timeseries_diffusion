import logging
import math

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class Diffusion(nn.Module):
    """
    Diffusion model for time series data.

    The model is initialized with a denoising network, a noise sampler and a way to compute Mahalanobis distances.
    Ususally, the noise sampler and the Mahalanobis distances will be based the same Gaussian Process.

    The noise schedule can be either linear, quadratic or cosine.
    """

    def __init__(
        self,
        network,
        diffusion_time_steps,
        noise_sampler,
        mal_dist_computer,
        schedule="linear",
        start_beta=1e-4,
        end_beta=0.02,
    ):
        super().__init__()
        assert network.signal_length == noise_sampler.signal_length
        assert network.signal_length == mal_dist_computer.signal_length

        self.device = "cpu"  # default
        self.network = network
        self.noise_sampler = noise_sampler
        self.mal_dist_computer = mal_dist_computer
        self.schedule = schedule
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.diffusion_time_steps = diffusion_time_steps

        if self.schedule == "linear":
            _betas = torch.linspace(start_beta, end_beta, diffusion_time_steps)
        elif self.schedule == "quad":  # also known as scaled linear
            _betas = (
                torch.linspace(start_beta**0.5, end_beta**0.5, diffusion_time_steps)
                ** 2.0
            )
        elif self.schedule == "cosine":
            _betas = cosine_betas(diffusion_time_steps)

        self.register_buffer("betas", _betas)
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("unormalized_probs", torch.ones(self.diffusion_time_steps))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]
        self.noise_sampler.to(*args, **kwargs)
        self.mal_dist_computer.to(*args, **kwargs)
        return self

    def train_batch(self, batch, cond=None, mask=None):
        self.train()
        batch_size = batch.shape[0]
        time_index = self.unormalized_probs.multinomial(
            num_samples=batch_size, replacement=True
        )
        train_alpha_bars = self.alpha_bars[time_index].unsqueeze(-1).unsqueeze(-1)
        noise = self.noise_sampler.sample(
            sample_shape=(
                batch_size,
                self.network.signal_channel,
                self.network.signal_length,
            )
        )
        assert noise.shape == batch.shape
        noisy_sig = (
            torch.sqrt(train_alpha_bars) * batch
            + torch.sqrt(1.0 - train_alpha_bars) * noise
        )
        res = self.network.forward(noisy_sig, time_index, cond=cond)
        diff = noise - res
        malhabonis = self.mal_dist_computer.sqrt_mal(diff)
        if mask is not None:
            log.debug("Masking loss is applied!")
            # Compute loss only on the observed samples
            malhabonis = malhabonis * mask
        return torch.einsum("icl,icl->i", malhabonis, malhabonis)

    def sample(
        self,
        num_samples,
        cond=None,
        sample_length=None,
        sampler=None,
        noise_type="alpha_beta",
        history=False,
    ):
        if sampler is None:
            sampler = self.noise_sampler
        if sample_length is None:
            sample_length = self.noise_sampler.signal_length

        if cond is not None:
            cond_batch, cond_channel, cond_length = cond.shape
            assert cond_batch == 1 or cond_batch == num_samples
            assert cond_length == sample_length
            if cond_batch == 1:
                cond = cond.repeat(num_samples, 1, 1)

        self.eval()
        with torch.no_grad():
            state = sampler.sample(
                sample_shape=(
                    num_samples,
                    self.network.signal_channel,
                    sample_length,
                )
            )

            history_ls = [state] if history else None

            for i in range(self.diffusion_time_steps):
                timestep = self.diffusion_time_steps - i - 1
                time_vector = torch.tensor([timestep], device=self.device).repeat(
                    num_samples
                )
                samp = sampler.sample(
                    sample_shape=(
                        num_samples,
                        self.network.signal_channel,
                        sample_length,
                    )
                )

                state = (1 / torch.sqrt(self.alphas[timestep])) * (
                    state
                    - (
                        (
                            (1.0 - self.alphas[timestep])
                            / (torch.sqrt(1.0 - self.alpha_bars[timestep]))
                        )
                        * self.network.forward(
                            state,
                            time_vector,
                            cond=cond,
                        )
                    )
                )

                if timestep > 0:
                    if noise_type == "beta":
                        sigma = self._get_beta(timestep)
                    elif noise_type == "alpha_beta":
                        sigma = self._get_alpha_beta(timestep)
                    state += sigma * samp

                if history:
                    history_ls.append(state)

            return state, history_ls

    # Mask: 1 -> sample is present, 0 sample is not present
    def impute(
        self, signal, mask, num_samples=None, cond=None, noise_type="alpha_beta"
    ):
        signal_batch, signal_channel, signal_length = signal.shape
        if num_samples is None:
            num_samples = signal_batch
        else:
            assert signal_batch == 1

        mask_batch, mask_channel, mask_length = mask.shape
        assert mask_batch == 1 or mask_batch == num_samples
        assert signal_channel == mask_channel
        assert signal_length == mask_length

        if cond is not None:
            cond_batch, cond_channel, cond_length = cond.shape
            assert cond_batch == 1 or cond_batch == num_samples
            assert signal_channel == cond_channel
            assert signal_length == cond_length
            if cond_batch == 1:
                cond = cond.repeat(num_samples, 1, 1)

        self.eval()
        with torch.no_grad():
            state = self.noise_sampler.sample(
                sample_shape=(
                    num_samples,
                    self.network.signal_channel,
                    self.network.signal_length,
                )
            )
            for i in range(self.diffusion_time_steps):
                timestep = self.diffusion_time_steps - i - 1
                time_vector = torch.tensor([timestep], device=self.device).repeat(
                    num_samples
                )
                samp = self.noise_sampler.sample(
                    sample_shape=(
                        num_samples,
                        self.network.signal_channel,
                        self.network.signal_length,
                    )
                )

                state_tild = (1 / torch.sqrt(self.alphas[timestep])) * (
                    state
                    - (
                        (
                            (1.0 - self.alphas[timestep])
                            / (torch.sqrt(1.0 - self.alpha_bars[timestep]))
                        )
                        * self.network.forward(
                            state,
                            time_vector,
                            cond=cond,
                        )
                    )
                )

                if timestep > 0:
                    if noise_type == "beta":
                        sigma = self._get_beta(timestep)
                    elif noise_type == "alpha_beta":
                        sigma = self._get_alpha_beta(timestep)
                    state_tild += sigma * samp

                noisy_samp = (
                    torch.sqrt(self.alpha_bars[timestep]) * signal
                    + torch.sqrt(1.0 - self.alpha_bars[timestep]) * samp
                )
                state_hat = (1.0 / self.alphas[timestep]) * (
                    noisy_samp
                    - (
                        self.betas[timestep]
                        / torch.sqrt(1.0 - self.alpha_bars[timestep])
                    )
                    * samp
                )

                state = mask * state_hat + (1.0 - mask) * state_tild
            return state

    def _get_beta(self, timestep):
        return self.betas[timestep] ** 0.5

    def _get_alpha_beta(self, timestep):
        if timestep == 0:
            return self._get_beta(timestep)
        return (
            ((1.0 - self.alpha_bars[timestep - 1]) / (1.0 - self.alpha_bars[timestep]))
            * self.betas[timestep]
        ) ** 0.5


class Trainer:
    """
    Training class for a diffusion model.

    Given a data loader and optimizer, it trains the model for one epoch.
    """

    def __init__(self, model, data_loader, optimizer, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.data_loader = data_loader

    def train_epoch(self):
        batchwise_losses = []
        for batch in self.data_loader:
            sig_batch = batch["signal"]
            batch_size = sig_batch.shape[0]
            sig_batch = sig_batch.to(self.model.device)
            # If a Dataloader provides these, use them. If not, don't.
            try:
                cond_batch = batch["cond"]
                cond_batch = cond_batch.to(self.model.device)
            except KeyError:
                cond_batch = None
            try:
                mask_batch = batch["mask"]
                mask_batch = mask_batch.to(self.model.device)
            except KeyError:
                mask_batch = None

            batch_loss = self.model.train_batch(
                sig_batch, cond=cond_batch, mask=mask_batch
            )
            batch_loss = torch.mean(batch_loss)

            batchwise_losses.append((batch_size, batch_loss.item()))

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        return batchwise_losses


def cosine_betas(num_diffusion_timesteps, max_beta=0.999):
    """
    Cosine beta schedule for diffusion model.

    Args:
        num_diffusion_timesteps: Number of diffusion timesteps.
        max_beta: Maximum beta value.

    Returns:
        Array of beta values.
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.from_numpy(np.array(betas, dtype=np.float32))
