#!/bin/sh

echo $1

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_conditional_brain \
    base.tag=$1_wn_conditional_brain_lconv_ddpm \
    dataset=tycho_conditional \
    diffusion=diffusion_linear_1000 \
    diffusion_kernel=white_noise \
    network=lconv_tycho \
    network.cond_channel=1 \
    optimizer=base_optimizer \
    optimizer.lr=0.0004 \
    optimizer.num_epochs=500 \
    +experiments/generate_samples=generate_samples \
    +experiments/fake_training_data_experiment=tycho_ftde \
    +experiments/likelihood_experiment=likelihood_experiment \
    +experiments/long_likelihood_experiment=long_likelihood_experiment

# Baseline
python3 ../ntd/utils/classifier_utils.py \
    base.experiment=$1_conditional_brain \
    base.tag=$1_conditional_brain_raw \
    dataset=tycho_conditional \
    classifier=tycho_classifier

echo "DONE"