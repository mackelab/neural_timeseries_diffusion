import logging
import os
import random
import time
from functools import partial

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf as OC
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset, random_split

from ntd.ajile_imputation_experiment import (
    conditional_ajile_imputation_experiment,
    imputation_experiment_loop,
)
from ntd.datasets import (
    AJILE_LFP,
    CRCNS_LFP,
    NER_BCI,
    TychoConditionalDataset,
    TychoUnconditionalDataset,
    load_ajile_data,
    load_ecog_data,
)
from ntd.diffusion_model import Diffusion, Trainer
from ntd.likelihood_computation import likelihood_experiment, long_likelihood_experiment
from ntd.networks import BaseConv, LongConv
from ntd.utils.classifier_utils import (
    TensorLabelDataset,
    train_and_test_classifier,
)
from ntd.utils.kernels_and_diffusion_utils import (
    OUProcess,
    WhiteNoiseProcess,
    generate_samples,
    single_channelwise_imputations,
)
from ntd.utils.utils import config_saver, count_parameters

log = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set the seed for all random number generators and switch to deterministic algorithms.
    This can hurt performance!

    Args:
        seed: The random seed.
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# cfg function
def init_diffusion_model(cfg):
    """Initialize the network, noise process and diffusion model.

    Args:
        cfg: The config object.

    Returns:
        diffusion: The diffusion model.
        network: The denoising network.
    """

    if cfg.diffusion_kernel.self == "white_noise":
        wn_process = WhiteNoiseProcess(
            cfg.diffusion_kernel.sigma_squared, cfg.dataset.signal_length
        )
        noise_sampler = wn_process
        mal_dist_computer = wn_process
    elif cfg.diffusion_kernel.self == "ou":
        ou_process = OUProcess(
            cfg.diffusion_kernel.sigma_squared,
            cfg.diffusion_kernel.ell,
            cfg.dataset.signal_length,
        )
        noise_sampler = ou_process
        mal_dist_computer = ou_process
    else:
        raise NotImplementedError

    diffusion_steps = cfg.diffusion.diffusion_steps
    if cfg.network.self == "bconv":
        network = BaseConv(
            signal_length=cfg.dataset.signal_length,
            signal_channel=cfg.network.signal_channel,
            time_dim=cfg.network.time_dim,
            cond_channel=cfg.network.cond_channel,
            kernel_size=cfg.network.kernel_size,
            hidden_channel=cfg.network.hidden_channel,
            norm_type=cfg.network.norm_type,
            activation_type=cfg.network.activation_type,
            padding_mode=cfg.network.padding_mode,
        )
    elif cfg.network.self == "lconv":
        network = LongConv(
            signal_length=cfg.dataset.signal_length,
            signal_channel=cfg.network.signal_channel,
            time_dim=cfg.network.time_dim,
            cond_channel=cfg.network.cond_channel,
            hidden_channel=cfg.network.hidden_channel,
            in_kernel_size=cfg.network.in_kernel_size,
            out_kernel_size=cfg.network.out_kernel_size,
            slconv_kernel_size=cfg.network.slconv_kernel_size,
            num_scales=cfg.network.num_scales,
            decay_min=cfg.network.decay_min,
            decay_max=cfg.network.decay_max,
            heads=cfg.network.heads,
            in_mask_mode=cfg.network.in_mask_mode,
            mid_mask_mode=cfg.network.mid_mask_mode,
            out_mask_mode=cfg.network.out_mask_mode,
            activation_type=cfg.network.activation_type,
            norm_type=cfg.network.norm_type,
            padding_mode=cfg.network.padding_mode,
            use_fft_conv=cfg.network.use_fft_conv,
        )
    else:
        raise NotImplementedError

    diffusion = Diffusion(
        network=network,
        noise_sampler=noise_sampler,
        mal_dist_computer=mal_dist_computer,
        diffusion_time_steps=diffusion_steps,
        schedule=cfg.diffusion.schedule,
        start_beta=cfg.diffusion.start_beta,
        end_beta=cfg.diffusion.end_beta,
    )
    return diffusion, network


# cfg function
def init_dataset(cfg):
    """
    Initialize the dataset.

    Args:
        cfg: The config object.

    Returns:
        train_data_set: The training dataset.
        test_data_set: The test dataset.
    """

    created_train_test_split = False
    if cfg.dataset.self == "tycho_unconditional":
        data_set = TychoUnconditionalDataset(
            signal_length=cfg.dataset.signal_length,
            channels=cfg.dataset.channels,
            session_one_start=cfg.dataset.session_one_start,
            session_one_end=cfg.dataset.session_one_end,
            filepath=cfg.dataset.filepath,
        )
    elif cfg.dataset.self == "tycho_conditional":
        data_set = TychoConditionalDataset(
            signal_length=cfg.dataset.signal_length,
            channels=cfg.dataset.channels,
            session_one_start=cfg.dataset.session_one_start,
            session_one_end=cfg.dataset.session_one_end,
            session_two_start=cfg.dataset.session_two_start,
            session_two_end=cfg.dataset.session_two_end,
            with_class_cond=cfg.dataset.with_class_cond,
            filepath=cfg.dataset.filepath,
        )
    elif cfg.dataset.self == "ner":
        data_set = NER_BCI(
            patient_id=cfg.dataset.patient_id,
            with_time_emb=cfg.dataset.with_time_emb,
            cond_time_dim=cfg.dataset.cond_time_dim,
            filepath=cfg.dataset.filepath,
        )
    elif cfg.dataset.self == "crcns":
        data_set = CRCNS_LFP(
            sessions=cfg.dataset.sessions,
            lfp_types=cfg.dataset.lfp_types,
            signal_length=cfg.dataset.signal_length,
            filepath=cfg.dataset.filepath,
        )
    elif cfg.dataset.self == "ajile":
        (
            X_train,
            y_train,
            mask_train,
            X_test,
            y_test,
            mask_test,
            train_mean,
            train_std,
        ) = load_ajile_data(
            cfg.dataset.pat_id,
            time_before_event=cfg.dataset.time_before_event,
            time_after_event=cfg.dataset.time_after_event,
            lower_q=cfg.dataset.lower_percentile,
            upper_q=cfg.dataset.upper_percentile,
            iqr_factor=cfg.dataset.iqr_factor,
            filepath=cfg.dataset.filepath,
        )
        train_data_set = AJILE_LFP(
            signal_length=cfg.dataset.signal_length,
            data_array=X_train,
            label_array=y_train,
            mask_array=mask_train,
            train_mean=train_mean,
            train_std=train_std,
            conditional_training=cfg.dataset.conditional_training,
        )
        train_data_set = Subset(train_data_set, list(range(len(train_data_set))))
        test_data_set = AJILE_LFP(
            signal_length=cfg.dataset.signal_length,
            data_array=X_test,
            label_array=y_test,
            mask_array=mask_test,
            train_mean=train_mean,
            train_std=train_std,
            conditional_training=cfg.dataset.conditional_training,
        )
        test_data_set = Subset(test_data_set, list(range(len(test_data_set))))
        created_train_test_split = True
    else:
        raise NotImplementedError

    if not created_train_test_split:
        train_size = int(len(data_set) * cfg.dataset.train_test_split)
        test_size = len(data_set) - train_size
        train_data_set, test_data_set = random_split(
            data_set,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(cfg.dataset.split_seed),
        )
        log.debug(train_data_set.indices)
        log.debug(test_data_set.indices)

    return train_data_set, test_data_set


### Experiments


# cfg function
def ajile_experiment_wrapper(cfg, diffusion, test_dataset, train_dataset):
    """
    Wrapper for the ajile experiment.

    Args:
        cfg: The config object.
        diffusion: The diffusion model.
        test_dataset: The test dataset.
        train_dataset: The train dataset.
    """

    param_ajile_exp = partial(
        conditional_ajile_imputation_experiment,
        batch_size=cfg.ajile_experiment.batch_size,
        rf_n_estimators=cfg.ajile_experiment.rf_n_estimators,
        rf_max_depth=cfg.ajile_experiment.rf_max_depth,
        n_folds=cfg.ajile_experiment.n_folds,
        rf_seed=cfg.ajile_experiment.rf_seed,
    )

    ajile_res = imputation_experiment_loop(
        imputation_experiment=param_ajile_exp,
        diffusion=diffusion,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        dropout_lvls=cfg.ajile_experiment.dropout_lvls,
        num_repeat=cfg.ajile_experiment.num_repeat,
        mask_seed=cfg.ajile_experiment.mask_seed,
    )

    # For logging
    for lvl in cfg.ajile_experiment.dropout_lvls:
        res_arr = ajile_res[str(lvl)]
        log.info(np.mean(res_arr, axis=0))
        log.info(np.mean(res_arr[:, 2] - res_arr[:, 1]))
        log.info(np.std(res_arr[:, 2] - res_arr[:, 1]))

    config_saver(
        file=ajile_res,
        filename="ajile_results.pkl",
        cfg=cfg,
    )


# cfg function
def arficial_train_data_wrapper(cfg, test_dataset, train_dataset, samples):
    """
    Wrapper for the fake training data experiment.

    Args:
        cfg: The config object.
        test_dataset: The test dataset.
        train_dataset: The train dataset.
        samples: The generated samples.
    """
    labels = torch.stack([dic["label"] for dic in train_dataset])

    _classifier, final_test_dict = train_and_test_classifier(
        train_dataset=TensorLabelDataset(samples, labels),
        test_dataset=test_dataset,
        in_channel=cfg.classifier.in_channel,
        hidden_channel=cfg.classifier.hidden_channel,
        kernel_size=cfg.classifier.kernel_size,
        num_runs=cfg.classifier.num_runs,
        train_batch_size=cfg.classifier.train_batch_size,
        test_batch_size=cfg.classifier.test_batch_size,
        learning_rate=cfg.classifier.learning_rate,
        weight_decay=cfg.classifier.weight_decay,
        epochs=cfg.classifier.epochs,
        wandb_mode=cfg.base.wandb_mode,
        wandb_experiment=cfg.base.experiment,
        wandb_tag=cfg.base.tag,
        wandb_project=cfg.base.wandb_project,
        wandb_entity=cfg.base.wandb_entity,
        wandb_dir=cfg.base.home_path,
    )

    config_saver(
        file=final_test_dict,
        filename="classifier_results.pkl",
        cfg=cfg,
    )


# cfg function
def swr_prediction_wrapper(cfg, diffusion, test_dataset):
    """
    Wrapper for the sharp-wave ripple prediction experiment.

    Args:
        cfg: The config object.
        diffusion: The diffusion model.
        test_dataset: The test dataset.
    """

    imputations_swr = single_channelwise_imputations(
        diffusion=diffusion,
        test_dataset=test_dataset,
        channel=cfg.swr_prediction_experiment.channel,
        batch_size=cfg.swr_prediction_experiment.batch_size,
    )

    config_saver(
        file=imputations_swr,
        filename="imputations.pkl",
        cfg=cfg,
    )


# cfg function
def likelihood_computation_wrapper(cfg, diffusion, test_dataset):
    """
    Wrapper for the likelihood computation experiment on the test set.

    Args:
        cfg: The config object.
        diffusion: The diffusion model.
        test_dataset: The test dataset.
    """

    likelihoods = likelihood_experiment(
        diffusion=diffusion,
        test_data=test_dataset,
        batch_size=cfg.likelihood_experiment.batch_size,
    )

    config_saver(
        file=likelihoods,
        filename="likelihoods.pkl",
        cfg=cfg,
    )


def long_likelihood_computation_wrapper(cfg, diffusion, train_dataset):
    """
    Wrapper for the likelihood computation experiment on the whole recording.

    Args:
        cfg: The config object.
        diffusion: The diffusion model.
        train_dataset: The train dataset.
    """

    session_one_path = os.path.join(cfg.dataset.filepath, "Session1")
    session_one = load_ecog_data(
        session_one_path,
        cfg.dataset.channels,
        cfg.dataset.signal_length,
        cfg.long_likelihood_experiment.session_one_start,
        cfg.long_likelihood_experiment.session_one_end,
    )
    session_two_path = os.path.join(cfg.dataset.filepath, "Session2")
    session_two = load_ecog_data(
        session_two_path,
        cfg.dataset.channels,
        cfg.dataset.signal_length,
        cfg.long_likelihood_experiment.session_two_start,
        cfg.long_likelihood_experiment.session_two_end,
    )

    long_likelihoods = long_likelihood_experiment(
        diffusion=diffusion,
        train_dataset=train_dataset,
        window_arr=np.concatenate([session_one, session_two], axis=0),
        batch_size=cfg.long_likelihood_experiment.batch_size,
    )

    config_saver(
        file=long_likelihoods,
        filename="long_likelihoods.pkl",
        cfg=cfg,
    )


# cfg function
def generate_samples_wrapper(cfg, diffusion, dataset):
    """
    Wrapper for the sample generation.
    If not specified otherwise, the generated number of samples matches the size of the provided dataset.

    Args:
        cfg: The config object.
        diffusion: The diffusion model.
        dataset: Either train or test dataset.
    """
    num_samples = (
        len(dataset)
        if cfg.generate_samples.num_samples == 0
        else cfg.generate_samples.num_samples
    )

    try:
        cond = torch.stack([dic["cond"] for dic in dataset])[:num_samples]
        cond = cond.to(diffusion.device)
        # If available, take the first num_samples from the dataset
    except KeyError:
        cond = None

    samples = generate_samples(
        diffusion=diffusion,
        total_num_samples=num_samples,
        batch_size=cfg.generate_samples.batch_size,
        cond=cond,
    )

    config_saver(
        file=samples,
        filename="samples.pkl",
        cfg=cfg,
    )

    # reuse samples downstream
    return samples


# cfg function
def run_experiment(cfg, diffusion, test_dataset, train_dataset):
    """
    Run the experiments specified in the config.

    Args:
        cfg: The config object.
        diffusion: The diffusion model.
        test_dataset: The test dataset.
        train_dataset: The train dataset.

    Returns:
        samples: The generated samples, if some were generated.
    """

    samples = None
    if OC.select(cfg, "generate_samples") is not None:
        samples = generate_samples_wrapper(cfg, diffusion, train_dataset)
    if OC.select(cfg, "ajile_experiment") is not None:
        ajile_experiment_wrapper(cfg, diffusion, test_dataset, train_dataset)
    if OC.select(cfg, "ftde_experiment") is not None:  # FakeTrainingDataExperiment
        assert OC.select(cfg, "generate_samples") is not None
        arficial_train_data_wrapper(cfg, test_dataset, train_dataset, samples)
    if OC.select(cfg, "swr_prediction_experiment") is not None:
        swr_prediction_wrapper(cfg, diffusion, test_dataset)
    if OC.select(cfg, "likelihood_experiment") is not None:
        likelihood_computation_wrapper(cfg, diffusion, test_dataset)
    if OC.select(cfg, "long_likelihood_experiment") is not None:
        long_likelihood_computation_wrapper(cfg, diffusion, train_dataset)
    return samples


# cfg function
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    """
    Main function call. Allows logging of exceptions.

    Args:
        cfg: The config object.
    """

    try:
        training_and_eval_pipeline(cfg)
    except Exception as e:
        log.exception(e)
        raise


def training_and_eval_pipeline(cfg):
    """
    Main function.

    Args:
        cfg: The config object.
    """

    log.info("Start diffusion training script")

    assert cfg.base.experiment is not None
    if cfg.base.tag is None:
        cfg.base.tag = cfg.base.experiment

    log.info(OC.to_yaml(cfg))
    log.info("Config correct? Abort if not.")
    time.sleep(10)
    log.info("Start run.")

    if cfg.base.seed is not None:
        set_seed(cfg.base.seed)

    if cfg.base.use_cuda_if_available and torch.cuda.is_available():
        device = torch.device("cuda")
        environ_kwargs = {"num_workers": 0, "pin_memory": True}
        log.info("Using CUDA")
    else:
        device = torch.device("cpu")
        environ_kwargs = {}
        log.info("Using CPU")

    train_data_set, test_data_set = init_dataset(cfg)

    train_loader = DataLoader(
        train_data_set,
        batch_size=cfg.optimizer.train_batch_size,
        shuffle=True,
        **environ_kwargs,
    )
    log.info(f"{len(train_data_set)}, {len(test_data_set)}")

    diffusion, network = init_diffusion_model(cfg)

    optimizer = optim.AdamW(
        network.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    trainer = Trainer(diffusion, train_loader, optimizer, device)

    wandb_conf = OC.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        mode=cfg.base.wandb_mode,
        project=cfg.base.wandb_project,
        entity=cfg.base.wandb_entity,
        config=wandb_conf,
        dir=cfg.base.home_path,
        group=cfg.base.experiment,
        name=cfg.base.tag,
    )

    run.summary["num_network_parameters"] = count_parameters(network)

    scheduler = MultiStepLR(
        optimizer,
        milestones=cfg.optimizer.scheduler_milestones,
        gamma=cfg.optimizer.scheduler_gamma,
    )

    for i in range(cfg.optimizer.num_epochs):
        batchwise_losses = trainer.train_epoch()

        epoch_loss = 0
        for batch_size, batch_loss in batchwise_losses:
            epoch_loss += batch_size * batch_loss
        epoch_loss /= len(train_data_set)

        wandb.log({"Train loss": epoch_loss})
        log.info(f"Epoch {i} loss: {epoch_loss}")

        scheduler.step()

    run.finish()
    log.info("Training done!")

    log.info("Saving cfg and model if specified.")

    config_saver(
        file=cfg,
        filename=f"config.pkl",
        cfg=cfg,
    )

    config_saver(
        file=diffusion.state_dict(),
        filename=f"models.pkl",
        cfg=cfg,
    )

    log.info("Generate samples and run experiment!")
    samples = run_experiment(cfg, diffusion, test_data_set, train_data_set)
    log.info("Evaluation done!")
    return diffusion, samples


if __name__ == "__main__":
    main()
