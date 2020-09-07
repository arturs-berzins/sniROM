## Introduction
This repository covers the main functionality and visualization routines used in [Standardized Non-Intrusive Reduced Order Modeling Using Different Regression Models With Application to Complex Flow Problems](https://arxiv.org/abs/2006.13706). In particular, the skewed lid driven cavity benchmark is used. However, as we emphasize the re-usability of model hyperparameters due to the standardization steps, the code is intended to be [easily transferable](#applying-to-your-dataset) to other compliant datasets.
The [dataset of the skewed lid driven cavity benchmark](dataset) is available to encourage comparability with other methods. The other two datasets used in the paper are much larger (around 10 and 3 GB, respectively), but are available upon request.


## Structure
```
├─ dataset/          # Skewed lid driven cavity dataset
├─ data/             # Training and testing data (untracked)
├─ models/           # Definitions of regression models
├─ visualization/    # Definition of mesh and .vtp outputs (untracked)
├─ config.py         # Configuration specifying the problem
├─ utils.py          # Utilities for loading and storing
├─ 01-05_*.py	     # Core functionality
├─ 06-08_*.py	     # Prediction and visualization
├─ 99_*.py           # Plots for debugging and building intuition
├─ requirements.txt  # Python dependencies
└─ README.md
```


## Setup
Python 3.6.8 with dependencies listed in [`requirements.txt`](requirements.txt) was used and tested on Linux, macOS and Windows.
It is highly recommended to use a virtual environment, for example in the project root:
```shell
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```
Minor versions of dependencies should not be ignored.
- The largest dependency, namely, `PyTorch` is not included in `requirements.txt` since it requires a more specific installation. Please see the installation guide [here](https://pytorch.org/get-started/locally/). Versions `1.0.1.post2` on Linux and `1.6.0+cpu` on Windows were tested succesfully. CUDA is theoretically supported in the implementation, but the ANN and the datasets are too small to benefit from this.
- `ray.tune` currently has only experimental support on Windows. If you can't get it to run properly on your machine (as indicated by errors during `python 02_tune.py`), it is possible to skip the tuning procedure altogether and instead use the near-optimal hyperparameter configuration identified in the paper. See the comment in [`02_tune.py`](02_tune.py) for more information.
- The visualization is done using [ParaView](https://www.paraview.org/download/), which has to be installed separately to view the `.vtp` files. If you wish to use another visualization pipeline, you will not need ParaView and can also remove `mayavi` (containing `tvtk.api`) from the requirements. In that case, you will need to modify the visualization scripts [`07`](07_visualize_predictions.py) and [`08`](08_visualize_bases.py) to your liking.

After a successful setup, simply run the scripts sequentially from the project root folder.

## Applying to your dataset
The code should be easily transferable to other datasets. The dataset must be provided in the format described in [`dataset/readme.md`](dataset/readme.md) and the [`config.py`](config.py) must be adjusted accordingly. This is enough to run the core scripts `01`-`05`.

If you also wish to visualize the predictions to ParaView, you will need to provide the mesh representation under [`visualization/`](visualization) and possibly modify script [`07`](07_visualize_predictions.py) depending on the specific problem.
