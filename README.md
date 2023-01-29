# Isaac Gym Environments for Legged Robots  - Modified for Robot Force Classifier Training - Data Collection Branch #
This repository provides the environment used to train ANYmal (and other robots) to walk on rough terrain using NVIDIA's Isaac Gym.
It includes all components needed for sim-to-real transfer: actuator network, friction & mass randomization, noisy observations and random pushes during training.  
**Maintainer**: John Seon Keun Yi  
**Affiliation**: Georgia Institute of Technology 
**Contact**: johnsk9595@gmail.com  

### Useful Links ###
Original Project website: https://leggedrobotics.github.io/legged_gym/

### CODE STRUCTURE ###
1. Each environment is defined by an env file (`legged_robot.py`) and a config file (`legged_robot_config.py`). The config file contains two classes: one conatianing all the environment parameters (`LeggedRobotCfg`) and one for the training parameters (`LeggedRobotCfgPPo`).  
2. Both env and config classes use inheritance.  
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward.  
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `envs/__init__.py`, but can also be done from outside of this repository.  

### Usage ###
1. Setup for data collection:
    - This branch is created to collect labeled robot data for (four) different feedback forces.
    - Look at `legged_robot.py` to change duration, interval, and direction of applied force. Currently, force is applied to the torso of the robot. 
    - To change walking speed: change `commands -> ranges -> lin_vel_x` in `legged_robot_config.py` for `{min, max}` forward walking speed.
    - `play.py` receives the applies force, applies labels based on pre-defined thresholds, and saves the data as files. 
        - You can spawn n robots to record n data at one run.
        - The recorded robot data + labels are saved as csv form. See code to define path.
2. Run robot(s) in environment:  
```python legged_gym/scripts/play.py --task=aliengo --num_envs=3```
    - `--task`: name of robot in simulation (we use `aliengo` for robot guide dog)
    - `--num_envs`: number of robots deployed in simulation
    - By default the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.

### Branches ###
- controller: code to run robot on one designated walking speed
- data_collection: code to collect labeled motion data
- multistate: code that contains multiple experts trained for each forward walking speed (0, 0.5, 1.0, 1.5m/s)
- controller_new: not used
Detailed explanation for each branch is in the README of each branch.