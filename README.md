# Generating realistic neurophysiological time series with denoising diffusion probabilistic models

This repository contains research code for the preprint [Generating realistic neurophysiological time series with denoising diffusion probabilistic models](https://www.biorxiv.org/content/10.1101/2023.08.23.554148v1).

It contains all the scripts to run the experiments presented in the paper.  
To run the experiments, first download the datasets (links in the paper) and change the `filepath` parameter in the dataset configuration files in `conf/dataset`. To save all outputs locally, set the `save_path` parameter in `conf/base` to the desired location. Experiment tracking and saving with [Weights & Biases](https://wandb.ai/site) is also supported, but disabled by default.  
Finally, use the shell scripts provided in `scripts` to run the experiments! 

Additionally, an example jupyter notebook `example_ner_notebook.ipynb`, where a diffusion model is trained to generate the BCI Challenge @ NER 2015 data, is provided together with the data.

Coming soon: Notebooks for plotting the figures!


## Usage

Install dependencies:
```shell
git clone https://github.com/mackelab/neural_timeseries_diffusion.git
cd neural_timeseries_diffusion
pip install -e .
```

Run shell scripts:
```shell
cd scripts
./<name_of_script>.sh
```

Run the example jupyter notebook:
- Make sure that the `jupyter` package is installed
- Open and execute the notebook


