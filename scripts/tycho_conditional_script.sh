#!/bin/sh

python3 ../ntd/train_diffusion_model.py \
    base.experiment=tycho_conditional_experiment \
    base.tag=conv_wn \
    dataset=tycho_conditional \
    diffusion=diffusion_linear_200 \
    diffusion_kernel=white_noise \
    network=ada_conv_tycho \
    optimizer=base_optimizer \
    optimizer.lr=0.0004 \
    optimizer.num_epochs=500 \
    +experiments/generate_samples=generate_samples \
    +experiments/fake_training_data_experiment=tycho_ftde \
    +experiments/likelihood_experiment=likelihood_experiment \
    +experiments/long_likelihood_experiment=long_likelihood_experiment

# Baseline
python3 ../ntd/utils/classifier_utils.py \
    base.experiment=tycho_conditional_experiment \
    base.tag=raw_classifier \
    dataset=tycho_conditional \
    classifier=tycho_classifier

echo "DONE"