from setuptools import find_packages, setup

NAME = "ntd"
DESCRIPTION = "diffusion models for neural time series data"
URL = "https://github.com/mackelab/neural_timeseries_diffusion"
AUTHOR = "J. Vetter"
REQUIRES_PYTHON = ">=3.8.0"

REQUIRED = [
    "wandb==0.14.2",
    "numpy>=1.22",
    "scikit-learn==1.0.2",
    "scipy>=1.10.0",
    "hydra-core==1.2.0",
    "torch>=1.13.1",
    "xarray==2022.11.0",
    "h5py==3.6.0",
    "matplotlib==3.7.1",
    "einops==0.5.0",
    "pandas==2.0.3",
    "mne==1.4.2",
]

EXTRAS = {
    "dev": [
        "autoflake",
        "black",
        "deepdiff",
        "flake8",
        "isort",
        "ipykernel",
        "jupyter",
        "pep517",
        "pytest",
        "pyyaml",
    ],
}

setup(
    name=NAME,
    version="0.1.0",
    description=DESCRIPTION,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["conf"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="AGPLv3",
)
