<div align="center">    
 
# Bio-Inspired Plastic Neural Networks for Zero-Shot Out-of-Distribution Generalization in Complex Animal-Inspired Robots


[![Paper](https://img.shields.io/badge/paper-arxiv.2007.02686-B31B1B.svg)](https://arxiv.org/abs/2503.12406)

</div>





This reposistory contains the code to train Hebbian networks on [Omniverse Isaac Gym Reinforcement Learning Environments for Isaac Sim](https://github.com/isaac-sim/OmniIsaacGymEnvs) as described in our paper [Bio-Inspired Plastic Neural Networks for Zero-Shot Out-of-Distribution Generalization in Complex Animal-Inspired Robots, 2025](https://arxiv.org/abs/2503.12406).





PLEASE NOTE: The version of OmniIsaacGymEnvs being used here is 4.0.0, which is the final release version of Isaac Gym.


## System Requirements
It is recommended to have at least 32GB RAM and a GPU with at least 12GB VRAM. 



## Installation

Follow the Isaac Sim [documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) to install the latest Isaac Sim release. 


*This repository has been tested on Isaac Sim version 4.0.0.*


Once installed, this repository can be used as a python module, `omniisaacgymenvs`, with the python executable provided in Isaac Sim.

To install `omniisaacgymenvs`, first clone this repository:

```bash
git clone https://github.com/worasuch/BioInspiredPlasticNeuralNets.git
```

Once cloned, locate the [python executable in Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html). By default, this should be `python.sh`. We will refer to this path as `PYTHON_PATH`.


To set a `PYTHON_PATH` variable in the terminal that links to the python executable, we can run a command that resembles the following. Make sure to update the paths to your local path.


```
For Linux: alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
For Windows: doskey PYTHON_PATH=C:\Users\user\AppData\Local\ov\pkg\isaac_sim-*\python.bat $*
For IsaacSim Docker: alias PYTHON_PATH=/isaac-sim/python.sh
```


Install `omniisaacgymenvs` as a python module for `PYTHON_PATH`:


```bash
PYTHON_PATH -m pip install -e .
```


The following error may appear during the initial installation. This error is harmless and can be ignored.

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
```

Finally, open the Isaac Sim GUI and import the robot model (e.g., gecko robot) to the following path: 
```omniverse://localhost/Projects/slalom/```

The robot model is provided in this repository under: ```BioInspiredPlasticNeuralNets/omniisaacgymenvs/robots/models/```

For example, the gecko-like robot model used is [*slalom_fixedbody_16dof.usd*](
https://github.com/worasuch/BioInspiredPlasticNeuralNets/blob/main/omniisaacgymenvs/robots/models/slalom_fixedbody_16dof.usd)

## How to run   

*Note: All commands should be executed from `OmniIsaacGymEnvs/omniisaacgymenvs`.* 


To train gecko-like robot (named Slalom), run:

```bash
PYTHON_PATH scripts/es_train.py task=Slalom num_envs=10 test=False headless=False
```


An Isaac Sim app window should be launched. Once Isaac Sim initialization completes, the robot scene will be constructed and simulation will start running automatically. The process will terminate once training finishes.


To achieve maximum performance, launch training in `headless` mode with a larger number of robot environments `num_envs` as follows:

```bash
PYTHON_PATH scripts/es_train.py task=Slalom num_envs=1024 test=False headless=True
```


## References

For more information, please refer to [Omniverse Isaac Gym Reinforcement Learning Environments for Isaac Sim](https://github.com/isaac-sim/OmniIsaacGymEnvs)