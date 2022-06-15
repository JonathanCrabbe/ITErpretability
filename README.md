# ITErpretability
Explainability in the CATE setting.


## Installation

```bash
pip install -r requirements.txt
```

## Running experiments 

You can run the experiments using the following commands: 

- Experiment 1: Altering the Strength of Predictive Effects

```bash
python run_experiments.py --experiment_name=predictive_sensitivity
```

- Experiment 2: Incorporating Nonlinearities

```bash
python run_experiments.py --experiment_name=nonlinearity_sensitivity
```

- Experiment 3: The Effect of Confounding

```bash
python run_experiments.py --experiment_name=propensity_sensitivity
```

The results from all experiments are saved in results/. You can then plot the results by running the code in the notebook plot_results.ipynb. 

Note that we use the PyTorch implementations of the different CATE learners from the catenets Python package: https://github.com/AliciaCurth/CATENets.
