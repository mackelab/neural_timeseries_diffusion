#!/bin/sh

echo $1

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_inria \
    base.tag=$1_inria_wn_lconv_recover \
    dataset=ner \
    diffusion=diffusion_linear_1000 \
    diffusion_kernel=white_noise \
    network=lconv_ner \
    optimizer=base_optimizer \
    optimizer.num_epochs=1000 \
    optimizer.lr=0.0004 \
    +experiments/generate_samples=generate_samples

echo "DONE"