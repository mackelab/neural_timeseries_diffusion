# Generating realistic neurophysiological time series with denoising diffusion probabilistic models

![Overview Figure](overview_figure.png)

Code for the paper ["Generating realistic neurophysiological time series with denoising diffusion probabilistic models"](https://www.cell.com/patterns/fulltext/S2666-3899(24)00189-2), (2024).

The repository contains all the scripts to run the experiments presented in the paper.  
To run the experiments, first download the datasets (links in the paper) and change the `filepath` parameter in the dataset configuration files in `conf/dataset`. To save all outputs locally, set the `save_path` parameter in `conf/base` to the desired location.

Experiment tracking and saving with [Weights & Biases](https://wandb.ai/site) is also supported, but disabled by default.  
Finally, use the shell scripts provided in `scripts` to run the experiments! 

Additionally, an example Jupyter notebook `example_ner_notebook.ipynb`, where a diffusion model is trained to generate the BCI Challenge @ NER 2015 data, is provided together with the data.

## Installation
Install dependencies via pip:
```shell
git clone https://github.com/mackelab/neural_timeseries_diffusion.git
cd neural_timeseries_diffusion
pip install -e .
```

## Further information

#### How can apply the DDPM to my datasets?
This requires writing a dataloader for your dataset. Example dataloaders (for all the datasets used in paper) can be found in 
`datasets.py`.

#### What options exist for the denoiser architectures?
Two denoiser different denoiser architectures based on [structured convolutions](https://arxiv.org/abs/2210.09298) are provided, differing only in the way conditional information is passed to the network. 

In general, the architecture using adaptive layer norm (`AdaConv`) is preferable and a good first choice over the one where conditional information is just concatenated (`CatConv`) (see [this paper](https://arxiv.org/abs/2212.09748) for a comparison of both approaches for an image diffusion model).

#### White noise vs. OU process
The standard white noise is a good initial choice. However, using the OU process can improve the quality of the generated power spectra. This requires setting the length scale of the OU process as an additional hyperparameter in the `diffusion_kernel` config files.

## Running the experiments
After downloading the datasets and updating the file paths, run the experiment shell scripts:
```shell
cd scripts
./<name_of_script>.sh
```
This produces result files which can then be passed to the plotting notebooks.