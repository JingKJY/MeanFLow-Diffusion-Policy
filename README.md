# Meanflow Diffusion Policy (MDP)

[English](README.md) | [简体中文](README_zh.md)

Meanflow Diffusion Policy (MDP) is a robot manipulation policy that combines MeanFlow action generation with Diffusion Policy observation encoders, conditioning modules, and training infrastructure. It supports point-cloud and low-dimensional observations across simulation and real-robot experiments.

MDP provides training and evaluation workflows for Push-T, Robomimic, non-prehensile manipulation, and real-robot tasks. The repository contains the policy implementation, experiment configurations, environment interfaces, simulation assets, and reusable training utilities.

## Included components

- MeanFlow policies for point-cloud, Push-T, critic, and real-robot experiments.
- Hydra configurations for Robomimic, Push-T, non-prehensile tilt, and real-robot tasks.
- Dataset adapters, simulation environments, environment runners, and real-robot ZMQ clients.
- Diffusion Policy encoders, normalizers, guider/critic modules, EMA, and shared utilities required by MDP.
- MuJoCo/Robosuite XML, mesh, and texture assets needed by the included simulation environments.

The Python import namespace is `Mix_diffusion_policy`. The repository and Python distribution are named **Meanflow Diffusion Policy**, abbreviated **MDP**.

## Installation

The code is intended for Linux with an NVIDIA GPU. The tested environment uses Ubuntu 20.04 and Python 3.9.

```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
conda env create -f conda_environment.yaml
conda activate mdp
pip install -e .
```

Some real-robot modules additionally require `pyzmq`, `open3d`, and a compatible `ur-analytic-ik` installation. Hardware dependencies are not needed for simulation-only use.

## Data layout

Datasets and checkpoints are not distributed with this repository. Place them under `data/` or override the corresponding Hydra fields from the command line.

```text
data/
├── robomimic/datasets/<task>/<dataset>/low_dim_abs_pcd.hdf5
├── nonprehensile/tilt_fast.h5
├── pusht/pusht_cchi_v7_replay.zarr
├── real/<experiment>/gello_dataset.h5
└── checkpoints/<experiment>/*.ckpt
```

Configuration files are under `Mix_diffusion_policy/config/Mean/`. Paths in those files are examples and may be overridden without editing them.

## Simulation

The generic entry point accepts any configuration from the included Mean directory:

```bash
python train_mdp.py \
  --config-dir=Mix_diffusion_policy/config/Mean \
  --config-name=Mix_pusht.yaml \
  training.seed=42 \
  training.device=cuda:0
```

Robomimic example:

```bash
python train_mdp.py \
  --config-dir=Mix_diffusion_policy/config/Mean \
  --config-name=Mix_square_ph.yaml \
  task.dataset.dataset_path=data/robomimic/datasets/square/ph/low_dim_abs_pcd.hdf5 \
  training.device=cuda:0
```

Change `training.debug=true` for a short smoke run. Outputs are controlled by each configuration's `hydra.run.dir` field.

## Real robot

Real-robot configurations are the files containing `_real` in their names. Before connecting hardware, override at least the dataset path, checkpoint paths, and robot server host:

```bash
python train_mdp.py \
  --config-dir=Mix_diffusion_policy/config/Mean \
  --config-name=Mix_cylinder_real_2.yaml \
  task.dataset.dataset_path=data/real/cylinder_place/gello_dataset.h5 \
  env_runner.env.env_client.host=192.168.1.10 \
  training.device=cuda:0
```

The checked-in host is `127.0.0.1` as a safe placeholder. Verify workspace limits, emergency-stop behavior, coordinate frames, observation/action dimensions, and network endpoints on your platform before enabling motion. Real-robot execution is hardware-specific and is provided as a research reference.

## Configuration groups

| Group | Configurations |
| --- | --- |
| Push-T | `Mix_pusht.yaml`, `Mix_pusht_critic.yaml` |
| Robomimic | can, lift, square, and tool-hang variants |
| Non-prehensile | `Mix_tilt.yaml` |
| Real robot | block stacking, cylinder insertion/placement/ring variants |

## Data and generated outputs

Large files are deliberately excluded. Do not commit datasets, checkpoints, W&B runs, generated videos, local Hydra outputs, or robot credentials. See `.gitignore` for the publication defaults.

## Acknowledgements

MDP builds on concepts and infrastructure from MeanFlow, Diffusion Policy, Hierarchical Diffusion Policy (HDP), Robomimic, and Robosuite. Please cite these projects and the associated MDP publication when bibliographic information becomes available.

## License

Released under the MIT License. Third-party packages and assets remain subject to their respective licenses.
