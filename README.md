# Uncertainty-Aware Trust Region Policy Optimization

This repository is the official implementation of the algorithm Uncertainty-Aware Trust Region Policy Optimization (UA-TRPO), which was introduced in the AAAI 2021 paper [Uncertainty-Aware Policy Optimization: A Robust, Adaptive Trust Region Approach](https://ojs.aaai.org/index.php/AAAI/article/view/17130).

UA-TRPO addresses the finite-sample estimation error present in both the gradient estimate and trust region estimate used in TRPO. By adapting to the level of uncertainty present in these estimates, UA-TRPO generates robust, stable performance even when updates must be made from limited data.

The version of UA-TRPO in this repository includes updates to the methodology originally proposed in the AAAI 2021 paper. See below for a list of changes.

Please consider citing our paper as follows:

```
@inproceedings{queeney_2021_uatrpo,
 author = {James Queeney and Ioannis Ch. Paschalidis and Christos G. Cassandras},
 title = {Uncertainty-Aware Policy Optimization: A Robust, Adaptive Trust Region Approach},
 booktitle = {Proceedings of the {AAAI} Conference on Artificial Intelligence},
 pages = {9377--9385},
 publisher = {{AAAI} Press},
 volume = {35},
 year = {2021}
}
```

## Requirements

The source code requires the following packages to be installed (we have included the latest version used to test the code in parentheses):

- python (3.8.13)
- dm-control (1.0.0)
- gurobi (9.5.1)
- gym (0.21.0)
- matplotlib (3.5.1)
- mujoco-py (1.50.1.68)
- numpy (1.22.3)
- scipy (1.8.0)
- seaborn (0.11.2)
- tensorflow (2.7.0)

See the file `environment.yml` for the latest conda environment used to run our code, which can be built with conda using the command `conda env create`.

Some OpenAI Gym environments and all DeepMind Control Suite environments require the MuJoCo physics engine. Please see the [MuJoCo website](https://mujoco.org/) for more information. 

Gurobi is only required to run generalized versions of TRPO and UA-TRPO, which incorporate sample reuse as proposed in [Queeney et al. (2022)](https://arxiv.org/abs/2206.13714). Generalized algorithms use Gurobi to determine the optimal policy weights for their theoretically supported sample reuse, which requires a Gurobi license. Please see the [Gurobi website](https://www.gurobi.com/downloads/) for more information on downloading Gurobi and obtaining a license. Alternatively, generalized versions of TRPO and UA-TRPO can be run without Gurobi by using uniform policy weights with the `--uniform` option.

## Training

Simulations can be run by calling `run` on the command line. See below for examples of running TRPO and UA-TRPO on both OpenAI Gym and DeepMind Control Suite environments:

```
python -m uatrpo.run --env_type gym --env_name HalfCheetah-v3 --alg_name trpo
python -m uatrpo.run --env_type gym --env_name HalfCheetah-v3 --alg_name uatrpo

python -m uatrpo.run --env_type dmc --env_name cheetah --task_name run --alg_name trpo
python -m uatrpo.run --env_type dmc --env_name cheetah --task_name run --alg_name uatrpo
```

Hyperparameters can be changed to non-default values by using the relevant option on the command line. For more information on the inputs accepted by `run`, use the `--help` option or reference `common/cmd_utils.py`.

In order to evaluate robustness, adversarial gradient noise can be added to the training process using the `--adversary_mult` option. This represents a multiple of standard error in each dimension of the gradient estimate, which determines the magnitude of adversarial noise applied to each dimension.

The results of simulations are saved in the `logs/` folder upon completion.

### What's New?

1. In the AAAI 2021 paper, updates are restricted to a subspace that is determined via random projections on the uncertainty-aware trust region matrix **M**. In this repository, we instead determine the subspace via random projections on the Fisher Information Matrix **F** *before* constructing the uncertainty-aware trust region matrix. By doing so, all uncertainty calculations can be made in the low-dimensional subspace, which has benefits in terms of computation and memory. 

2. Calculations of (i) the covariance matrix used in UA-TRPO and (ii) adversarial gradient noise can be made using minibatch gradients to improve computation and memory. This can be done by choosing the number of minibatch gradients to consider with the inputs `--ua_nbatch` and `--adversary_nbatch`, respectively. Gradients can instead be calculated for every sample (i.e., minibatches of size one) by setting these inputs to `0`. The calculation of the adaptive trust region coefficient uses the number of (minibatch) samples rather than the number of trajectories, and the default value of `c` has been updated to reflect this change.

3. The repository supports generalized versions of TRPO and UA-TRPO by using `--alg_name getrpo` and `--alg_name geuatrpo`, respectively. These algorithms incorporate sample reuse using the methods proposed in [Queeney et al. (2022)](https://arxiv.org/abs/2206.13714). Sample reuse is based on the on-policy batch size given by `Bn`, where `n` represents the minimum possible batch size. This can be updated using the inputs `--B` and `--n`. By default, `B=1` which reduces to the on-policy case with batch size given by `n`.

## Evaluation

The results of simulations saved in the `logs/` folder can be visualized by calling `plot` on the command line and passing file names to the input `--import_files`:

```
python -m uatrpo.plot --import_files <filename_1> <filename_2> ...
```

By default, this command saves a plot of average performance throughout training in the `figs/` folder. Other metrics can be plotted using the `--metric` option. For more information on the inputs accepted by `plot`, use the `--help` option or reference `common/plot_utils.py`.