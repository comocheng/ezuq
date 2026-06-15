# ezuq
Put error bars on your microkinetic model.


## What is ezuq?
It's a Python package to help you run uncertainty quantification on a microkinetic model. The examples use combustion-type experiments: ignition delays (rapid compression machine/shock tube), flame speeds (burner), and species concentrations (jet-stirred reactor). This package helps you propagate uncertainty through your model to put error bars on the outputs like the jet-stirred reactor example below:
![JSR concentration with error bars](https://raw.githubusercontent.com/comocheng/ezuq/main/examples/00_uncorrelated_ethane/monte_carlo/unreduced_mc_results.png)


## Installation
1. Clone Repo
2. `pip install -e .`


TODO - pypi pip install and conda install


## How it works
1. Provide a microkinetic model with covariance matrices of the species/reaction uncertainties.
2. Configure your reactor simulation.
3. Run initial screening to find the most important parameters.
4. Run global sampling to estimate model uncertainty.
5. Make nice plots of the results.
