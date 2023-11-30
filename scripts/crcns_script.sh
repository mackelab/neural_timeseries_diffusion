#!/bin/sh

python3 ../ntd/train_diffusion_model.py \
    base.experiment=crcns_experiment \
    base.tag=lconv_wn \
    dataset=crcns \
    diffusion=diffusion_quad_500 \
    diffusion_kernel=white_noise \
    network=lconv_crcns \
    optimizer=base_optimizer \
    optimizer.lr=0.0001 \
    optimizer.num_epochs=500 \
    +experiments/generate_samples=generate_samples \
    +experiments/swr_prediction_experiment=swr_prediction_experiment \
    generate_samples.batch_size=1000

echo "DONE"