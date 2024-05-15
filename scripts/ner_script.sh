#!/bin/sh

python3 ../ntd/train_diffusion_model.py \
    base.experiment=ner_experiment \
    base.tag=lconv_wn \
    dataset=ner \
    diffusion=diffusion_linear_200 \
    diffusion_kernel=white_noise \
    network=lconv_ner \
    optimizer=base_optimizer \
    optimizer.num_epochs=1000 \
    optimizer.lr=0.0004 \
    +experiments/generate_samples=generate_samples

echo "DONE"