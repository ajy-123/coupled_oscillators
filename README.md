# Differentiating Neuron Ring Oscillators

This repository contains the code used to generate the data sets for Reservoir Computation with Networks of Differentiating Neuron Ring Oscillators, along with some images and Jupyter notebooks used for analysis.

## Environment Setup

To run the Python scripts in this repository, you need to set up an environment with the appropriate dependencies. To get started, clone our repository and navigate to its directory:

```bash
   git clone https://github.com/ajy-123/coupled_oscillators.git
   cd coupled_oscillators
```
### Prerequisites

This project was developed and tested using **Python 3.10.8**. Please ensure you have this version (or a compatible version) installed on your system.

### Required Libraries

The following Python libraries are required to run the the Python scripts in the scripts folder:

- `numpy`
- `networkx`
- `scipy`
- `keras`
- `numpy-hilbert-curve`

Additional plotting libraries were used to generate plots:
- `matplotlib`
- `seaborn`

To set up your environment, ensure that you install all of these packages individually, or via requirements.txt:
```bash
pip install -r requirements.txt
```

It is recommended that you do the envrionment setup within a Python virtual envrionment.

## Organization 

This repository contains three main folders: scripts, notebooks, and imgs.

scripts contains the Python and shell scripts that were used to generate the datasets later used for reservoir computation. The full data sets are available upon request. This directory also contains a subdirectory *prev* which contains scripts that were used in earlier work but did not ultimately impact the final results. The main files within script that would be useful in recreation of ourr experiments are:

1. `simulation.py` contains all of the classes and methods needed to run a simulation through our reservoir structure. It can be modified as needed and with different inputs to suit new experiments.
2. `make_datasets.py` and its associated `make_datasets.sh` were used to document the reservoir outputs specifically for the MNIST digit recognition dataset. `make_datasets.sh` can be used accordingly to produce results for new sizes or parameters.
3. `param_tuning.py` and its assocaited `param_tuning.sh` were used to document the reservoir outputs for MNIST digit recognition under different hyperparameters of *p* and *eps* discussed in our paper. `param_tuning.sh` can be used to produce new datasets for different hyperparameters.

imgs contains all of the images generated for our final results, as well as some earlier ones used for exploratory analysis.

notebooks contains Jupyter notebooks which contain analysis, visualization, and other plots.