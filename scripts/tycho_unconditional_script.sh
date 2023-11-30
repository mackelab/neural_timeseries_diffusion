#!/bin/sh

python3 ../ntd/train_diffusion_model.py \
    base.experiment=tycho_unconditional_experiment \
    base.tag=lconv_ou \
    dataset=tycho_unconditional \
    diffusion=diffusion_quad_500 \
    diffusion_kernel=ou10 \
    network=lconv_tycho \
    network.in_mask_mode=restricted \
    network.mid_mask_mode=off_diag_4 \
    network.out_mask_mode=restricted \
    network.signal_channel=128 \
    network.hidden_channel=16 \
    optimizer=base_optimizer \
    optimizer.lr=0.0004 \
    optimizer.num_epochs=250 \
    +experiments/generate_samples=generate_samples

echo "DONE"