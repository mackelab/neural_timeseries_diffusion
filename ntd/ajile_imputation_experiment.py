import logging

import numpy as np
import torch
from einops import repeat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from ntd.utils.kernels_and_diffusion_utils import generate_samples

log = logging.getLogger(__name__)


def conditional_ajile_imputation_experiment(
    diffusion,
    train_dataset,
    test_dataset,
    batch_size=133,
    percent_dropout_channels=0.5,
    mask_seed=0,
    rf_n_estimators=240,
    rf_max_depth=9,
    rf_seed=0,
    n_folds=3,
):
    """
    Runs the AJILE imputation experiment for a single patient given a trained diffusion model
    and the corresponding train and test datasets.

    Args:
        diffusion: Trained diffusion model
        train_dataset: Train dataset
        test_dataset: Test dataset
        batch_size: Batch size for diffusion model
        percent_dropout_channels: Percent of channels to drop
        mask_seed: Seed for random number generator for channel dropout
        rf_n_estimators: Number of estimators for random forest
        rf_max_depth: Max depth for random forest
        rf_seed: Seed for random number generator for random forest
        n_folds: Number of folds for random forest training and evalutation

    Returns:
        Tuple of
            mean full test accuracy,
            mean zero imputation accuracy,
            mean diffusion imputation accuracy
    """

    full_train_signals = torch.stack([dic["signal"] for dic in train_dataset])
    full_train_labels = torch.cat([dic["label"] for dic in train_dataset])
    num_train_samples, num_channels, signal_length = full_train_signals.shape

    full_test_signals = torch.stack([dic["signal"] for dic in test_dataset])
    full_test_labels = torch.cat([dic["label"] for dic in test_dataset])

    percent_cond_channels = 1.0 - percent_dropout_channels
    num_cond_channels = int(np.round(percent_cond_channels * num_channels))

    rng = np.random.default_rng(mask_seed)
    cond_channel = list(rng.choice(num_channels, size=num_cond_channels, replace=False))
    print(cond_channel)

    imputation_mask = torch.zeros(num_channels, signal_length)
    imputation_mask[cond_channel, :] = 1.0

    imputation_mask_repeat = repeat(
        imputation_mask, "c t -> b c t", b=full_test_signals.shape[0]
    )
    full_test_cond = torch.cat(
        [imputation_mask_repeat, imputation_mask_repeat * full_test_signals], dim=1
    )

    X_test_cond_gen = generate_samples(
        diffusion=diffusion,
        total_num_samples=len(full_test_cond),
        batch_size=batch_size,
        cond=full_test_cond,
    )
    X_test_cond_gen = X_test_cond_gen.cpu()
    # First mask (:num_channels), then signal (num_channels:)!
    X_test_imputed = (
        X_test_cond_gen * (1.0 - full_test_cond[:, :num_channels, :])
        + full_test_cond[:, num_channels:, :]
    )

    train_mean, train_std = train_dataset.dataset.get_train_mean_and_std()

    # Unstandardize imputed signal
    X_test_imputed_uns = (X_test_imputed * train_std) + train_mean

    # Recover standardized full signals
    full_train_signals_uns = (full_train_signals * train_std) + train_mean
    full_test_signals_uns = (full_test_signals * train_std) + train_mean
    # Zero out channels in recovered signal
    mean_test_signals_uns = (
        full_test_signals_uns * full_test_cond[:, :num_channels, :]
        + (1.0 - full_test_cond[:, :num_channels, :]) * train_mean
    )

    def rf_np_reshape(tensor):
        return tensor.numpy().reshape(tensor.shape[0], -1)

    # Relevant parameters used in Brunton et al.:
    # n_estimators=240, max_depth=9, n_folds = 3

    log.info(f"Start RF training and eval on {n_folds} folds")

    full_test_accs = np.zeros(n_folds)
    full_mean_accs = np.zeros(n_folds)
    full_impu_accs = np.zeros(n_folds)
    rng_rf = np.random.default_rng(mask_seed)
    train_perm = rng_rf.permutation(num_train_samples)
    fold_size = num_train_samples // n_folds
    for idx in range(n_folds):
        leave_out = train_perm[idx * fold_size : (idx + 1) * fold_size]
        # If n_folds == 1, there is no validation set and the training set consists of all indices
        take_for_train = (
            np.setdiff1d(train_perm, leave_out) if n_folds > 1 else leave_out
        )

        model = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            class_weight="balanced",
            oob_score=True,
            random_state=rf_seed + idx,
        )
        clf = model.fit(
            rf_np_reshape(full_train_signals_uns)[take_for_train],
            full_train_labels.numpy()[take_for_train],
        )

        full_test_accuracy = accuracy_score(
            full_test_labels.numpy(), clf.predict(rf_np_reshape(full_test_signals_uns))
        )
        full_test_accs[idx] = full_test_accuracy

        mean_fill_test_accuracy = accuracy_score(
            full_test_labels.numpy(),
            clf.predict(rf_np_reshape(mean_test_signals_uns)),
        )
        full_mean_accs[idx] = mean_fill_test_accuracy

        imputation_test_accuracy = accuracy_score(
            full_test_labels.numpy(),
            clf.predict(rf_np_reshape(X_test_imputed_uns)),
        )
        full_impu_accs[idx] = imputation_test_accuracy

    log.info(full_test_accs)
    log.info(full_mean_accs)
    log.info(full_impu_accs)

    return (
        np.mean(full_test_accs),
        np.mean(full_mean_accs),
        np.mean(full_impu_accs),
    )


def imputation_experiment_loop(
    imputation_experiment,
    diffusion,
    train_dataset,
    test_dataset,
    dropout_lvls=[0.5, 0.7, 0.9],
    num_repeat=5,
    mask_seed=0,
):
    """
    Provides a loop to repeatedly run the imputation experiment for different dropout levels.

    Args:
        imputation_experiment: Partial imputation experiment function
        diffusion: Trained diffusion model
        train_dataset: Train dataset
        test_dataset: Test dataset
        dropout_lvls: Dropout levels to run the experiment for
        num_repeat: Number of times to repeat the experiment for each dropout level
        mask_seed_start: Seed for random number generator for channel dropout

    Returns:
        Dictionary of results for each dropout level and repetition
    """

    res_dict = {}
    for dropout_lvl in dropout_lvls:
        res_arr = np.zeros((num_repeat, 3))
        for idx in range(num_repeat):
            full_res, zero_res, impu_res = imputation_experiment(
                diffusion=diffusion,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                percent_dropout_channels=dropout_lvl,
                mask_seed=mask_seed + idx,
            )
            res_arr[idx, 0] = full_res
            res_arr[idx, 1] = zero_res
            res_arr[idx, 2] = impu_res
        res_dict[str(dropout_lvl)] = res_arr
    return res_dict
