# [DQL-RHC] ***D***iffusion Policy with Behaviour Cloning and ***Q*** ***L***earning in ***R***eceding ***H***orizon *C*ontrol mode

## DQL-RHC Overview
![Overview](/assets/method.jpg)
The overall architecture of DQL-RHC and the information flow between its components. The framework consists of four main components: the long dataset, the dynamic predictor, the critic, and the actor. The long dataset stores sequence-wise historical $\{S, S', A, R, \text{DONE}\}$ transitions, which are used by the dynamic predictor to generate future state sequence and the corresponding predicted reward sequence. These predicted values are then processed by the critic to estimate Q values, while the actor utilizes these Q values to generate optimal actions. The red-colored action sequence represent the planned actions over the action horizon.

## Installation

To install the repository, follow the steps below:

- Clone this repository: `git clone https://github.com/idonotlikemondays/DQL-RHC.git`
- Follow the [ManiSkill3 installation guide](https://github.com/haosulab/ManiSkill) to set up the required dependencies.

## Baseline Support

We modified several baseline algorithms, including BC-based architectures, offline RL architectures, and BC-RL hybrid architectures, for a comprehensive performance comparison with our proposed **DQL-RHC**. The specific baselines include:

- **BC-based methods**:
    - **BC** (Behaviour Cloning)
    - **DP-single** (Diffusion Policy in single step design)
    - [**DP-RHC**](https://github.com/real-stanford/diffusion_policy) (Diffusion Policy)
- **Offline RL methods**:
    - **SAC** (Soft Actor-Critic)
    - **PPO** (Proximal Policy Optimization)
- **BC-RL hybrid methods**:
    - [**TD3+BC**](https://github.com/sfujim/TD3_BC)
    - [**DQL**](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL) (Diffusion Policies for Offline RL)
    - **★** **DQL-RHC ★**

## Installation and Implementation

- Environment Creation: `conda create -n maniskill python=3.9`
- ManiSkill3 and dependencies Installation:
    
    ```python
    pip install --upgrade mani_skill # we use mani_skill==3.0.0b10
    
    # install a version of torch that is compatible with your system
    # we run with `pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1` in our work
    pip install torch
    
    # install other dependencies
    pip install wandb==0.13.3
    pip install gymnasium==0.29.1
    ```
    
- Demonstration Download:
    
    ```python
    # with no args this prints all available datasets
    python -m mani_skill.utils.download_demo
    
    # download the demonstration dataset for certain task
    python -m mani_skill.utils.download_demo 'PushCube-v1'
    python -m mani_skill.utils.download_demo 'StackCube-v1'
    python -m mani_skill.utils.download_demo 'PickCube-v1'
    python -m mani_skill.utils.download_demo 'PegInsertionSide-v1'
    ...
    # download the full datasets
    python -m mani_skill.utils.download_demo all
    ```
    
- Trajectories Preprocessing:
    
    ```python
    # for ee control mode
    # for other tasks just change the task name "PickCube-v1" in second row
    # for other control mode just change the control mode "pd_ee_delta_pos" in third row
    python -m mani_skill.trajectory.replay_trajectory \
      --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
      --use-first-env-state -c pd_ee_delta_pos -o state \
      --save-traj --num-procs 10 -b cpu 
      --record-rewards True --reward-mode="normalized_dense"
      
    # for joints control mode
    python -m mani_skill.trajectory.replay_trajectory \
      --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
      --use-first-env-state -c pd_joint_delta_pos -o state \
      --save-traj --num-procs 10 -b cpu 
      --record-rewards True --reward-mode="normalized_dense"
    ```
    
- Training:
    
    ```python
    # for ee control mode
    python dql_rhc.py --env-id "PickCube-v1" \
    	--demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.cpu.h5 \
    	--control-mode "pd_ee_delta_pos" --sim-backend "cpu" \ 
    	--max-episode-steps 100  --total-iters 30000 --num_demos 100 \ 
    	--seed 1 --track
    	
    # for joints control mode
    python dql_rhc.py --env-id "PickCube-v1" \ 
    	--demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.cpu.h5 \ 
    	--control-mode "pd_joint_delta_pos" --sim-backend "cpu" \ 
    	--max_episode_steps 100 --total-iters 30000 --num_demos 100 \ 
    	--seed 1 --track
    ```