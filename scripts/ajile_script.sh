#!/bin/sh

echo $1

kernel="ou10"
lr=0.001
num_epochs=500

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_ajile_experiment \
    base.tag=$1_lconv_ou_ec01 \
    dataset=ajile \
    dataset.pat_id=EC01 \
    diffusion=diffusion_quad_50 \
    diffusion_kernel=${kernel} \
    network=lconv_ajile \
    optimizer=base_optimizer \
    optimizer.lr=${lr} \
    optimizer.num_epochs=${num_epochs} \
    network.signal_channel=94 \
    network.cond_channel=188 \
    +experiments/ajile_experiment=ajile_experiment

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_ajile_experiment \
    base.tag=$1_lconv_ou_ec02 \
    dataset=ajile \
    dataset.pat_id=EC02 \
    diffusion=diffusion_quad_50 \
    diffusion_kernel=${kernel} \
    network=lconv_ajile \
    optimizer=base_optimizer \
    optimizer.lr=${lr} \
    optimizer.num_epochs=${num_epochs} \
    network.signal_channel=86 \
    network.cond_channel=172 \
    +experiments/ajile_experiment=ajile_experiment

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_ajile_experiment \
    base.tag=$1_lconv_ou_ec03 \
    dataset=ajile \
    dataset.pat_id=EC03 \
    diffusion=diffusion_quad_50 \
    diffusion_kernel=${kernel} \
    network=lconv_ajile \
    optimizer=base_optimizer \
    optimizer.lr=${lr} \
    optimizer.num_epochs=${num_epochs} \
    network.signal_channel=80 \
    network.cond_channel=160 \
    +experiments/ajile_experiment=ajile_experiment

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_ajile_experiment \
    base.tag=$1_lconv_ou_ec04 \
    dataset=ajile \
    dataset.pat_id=EC04 \
    diffusion=diffusion_quad_50 \
    diffusion_kernel=${kernel} \
    network=lconv_ajile \
    optimizer=base_optimizer \
    optimizer.lr=${lr} \
    optimizer.num_epochs=${num_epochs} \
    network.signal_channel=84 \
    network.cond_channel=168 \
    +experiments/ajile_experiment=ajile_experiment

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_ajile_experiment \
    base.tag=$1_lconv_ou_ec05 \
    dataset=ajile \
    dataset.pat_id=EC05 \
    diffusion=diffusion_quad_50 \
    diffusion_kernel=${kernel} \
    network=lconv_ajile \
    optimizer=base_optimizer \
    optimizer.lr=${lr} \
    optimizer.num_epochs=${num_epochs} \
    network.signal_channel=106 \
    network.cond_channel=212 \
    +experiments/ajile_experiment=ajile_experiment

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_ajile_experiment \
    base.tag=$1_lconv_ou_ec06 \
    dataset=ajile \
    dataset.pat_id=EC06 \
    diffusion=diffusion_quad_50 \
    diffusion_kernel=${kernel} \
    network=lconv_ajile \
    optimizer=base_optimizer \
    optimizer.lr=${lr} \
    optimizer.num_epochs=${num_epochs} \
    network.signal_channel=80 \
    network.cond_channel=160 \
    +experiments/ajile_experiment=ajile_experiment

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_ajile_experiment \
    base.tag=$1_lconv_ou_ec07 \
    dataset=ajile \
    dataset.pat_id=EC07 \
    diffusion=diffusion_quad_50 \
    diffusion_kernel=${kernel} \
    network=lconv_ajile \
    optimizer=base_optimizer \
    optimizer.lr=${lr} \
    optimizer.num_epochs=${num_epochs} \
    network.signal_channel=64 \
    network.cond_channel=128 \
    +experiments/ajile_experiment=ajile_experiment

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_ajile_experiment \
    base.tag=$1_lconv_ou_ec08 \
    dataset=ajile \
    dataset.pat_id=EC08 \
    diffusion=diffusion_quad_50 \
    diffusion_kernel=${kernel} \
    network=lconv_ajile \
    optimizer=base_optimizer \
    optimizer.lr=${lr} \
    optimizer.num_epochs=${num_epochs} \
    network.signal_channel=92 \
    network.cond_channel=184 \
    +experiments/ajile_experiment=ajile_experiment

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_ajile_experiment \
    base.tag=$1_lconv_ou_ec09 \
    dataset=ajile \
    dataset.pat_id=EC09 \
    diffusion=diffusion_quad_50 \
    diffusion_kernel=${kernel} \
    network=lconv_ajile \
    optimizer=base_optimizer \
    optimizer.lr=${lr} \
    optimizer.num_epochs=${num_epochs} \
    network.signal_channel=126 \
    network.cond_channel=252 \
    +experiments/ajile_experiment=ajile_experiment

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_ajile_experiment \
    base.tag=$1_lconv_ou_ec10 \
    dataset=ajile \
    dataset.pat_id=EC10 \
    diffusion=diffusion_quad_50 \
    diffusion_kernel=${kernel} \
    network=lconv_ajile \
    optimizer=base_optimizer \
    optimizer.lr=${lr} \
    optimizer.num_epochs=${num_epochs} \
    network.signal_channel=126 \
    network.cond_channel=252 \
    +experiments/ajile_experiment=ajile_experiment

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_ajile_experiment \
    base.tag=$1_lconv_ou_ec11 \
    dataset=ajile \
    dataset.pat_id=EC11 \
    diffusion=diffusion_quad_50 \
    diffusion_kernel=${kernel} \
    network=lconv_ajile \
    optimizer=base_optimizer \
    optimizer.lr=${lr} \
    optimizer.num_epochs=${num_epochs} \
    network.signal_channel=106 \
    network.cond_channel=212 \
    +experiments/ajile_experiment=ajile_experiment

python3 ../ntd/train_diffusion_model.py \
    base.experiment=$1_ajile_experiment \
    base.tag=$1_lconv_ou_ec12 \
    dataset=ajile \
    dataset.pat_id=EC12 \
    diffusion=diffusion_quad_50 \
    diffusion_kernel=${kernel} \
    network=lconv_ajile \
    optimizer=base_optimizer \
    optimizer.lr=${lr} \
    optimizer.num_epochs=${num_epochs} \
    network.signal_channel=116 \
    network.cond_channel=232 \
    +experiments/ajile_experiment=ajile_experiment