ALGO_NAME = 'BC_Diffusion_state_UNet'

import argparse
import os
import random
import copy
from distutils.util import strtobool
import time
import datetime
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from diffusion_policy.evaluate import evaluate
from mani_skill.utils import gym_utils
from mani_skill.utils.registration import REGISTERED_ENVS

from collections import defaultdict
from tqdm import tqdm

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from diffusion_policy.utils import IterationBasedBatchSampler, worker_init_fn
from diffusion_policy.make_env import make_eval_envs
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from dataclasses import dataclass, field
from typing import Optional, List
import tyro

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the actor performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v0"
    """the id of the environment"""
    demo_path: str = 'data/ms2_official_demos/rigid_body/PegInsertionSide-v0/trajectory.state.pd_ee_delta_pose.h5'
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 128
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    actor_lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    critic_lr: float = 1e-4
    """the learning rate of the critic"""
    obsPredictor_lr: float = 1e-4
    """the learning rate of the observation predictor"""
    obs_horizon: int = 2 # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8 # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = 16 # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    diffusion_step_embed_dim: int = 64 # not very important
    num_diffusion_iters: int = 5 # diffusion steps
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256]) # default setting is about ~4.5M params
    n_groups: int = 8 # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila
    grad_norm: int = 1
    """Clips gradient norm of an iterable of parameters"""

    # Other settings
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """the coefficient used for update target critic with weight of critic"""
    alpha: float = 2.5
    """normalize Q value in policy training, lambda = alpha/(|Q|.mean())"""
    beta: float = 1.0
    """regularize bc loss in policy training, actor_loss = q_loss + beta * bc_loss"""


    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning actors can learn faster."""
    critic_warmup_steps: int = 0
    """before this specific timestep, train Critic with ground truth data (true obs and true action)"""
    log_freq: int = 100
    """the frequency of logging the training metrics"""
    eval_freq: int = 1000
    """the frequency of evaluating the actor on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the actor on"""
    num_eval_envs: int = 8
    """the number of parallel environments to evaluate the actor on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = 'pd_joint_delta_pos'
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


TARGET_KEY_TO_SOURCE_KEY = {
    'states': 'env_states',
    'observations': 'obs',
    'success': 'success',
    'next_observations': 'obs',
    # 'dones': 'dones', # no 'dones' keyword in trajectory dataset
    'terminated': 'terminated',
    'truncated': 'truncated', 
    'rewards': 'rewards',
    'actions': 'actions',
}

from diffusion_policy.utils import load_traj_hdf5
def load_demo_dataset(path, keys=['observations', 'actions'], num_traj=None, concat=True):
    # assert num_traj is None
    raw_data = load_traj_hdf5(path, num_traj)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    _traj = raw_data['traj_0']
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in traj_0: {_traj.keys()}"
    dataset = {}
    for target_key in keys:
        # if 'next' in target_key:
        #     raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [ raw_data[idx][source_key] for idx in raw_data ]
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ['observations', 'states'] and \
                    len(dataset[target_key][0]) > len(raw_data['traj_0']['actions']):
                dataset[target_key] = np.concatenate([
                    t[:-1] for t in dataset[target_key]
                ], axis=0)
            elif target_key in ['next_observations', 'next_states'] and \
                    len(dataset[target_key][0]) > len(raw_data['traj_0']['actions']):
                dataset[target_key] = np.concatenate([
                    t[1:] for t in dataset[target_key]
                ], axis=0)
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            print('Load', target_key, dataset[target_key].shape)
        else:
            print('Load', target_key, len(dataset[target_key]), type(dataset[target_key][0]))
    return dataset


class SmallDemoDataset_DiffusionPolicy(Dataset): # Load everything into GPU memory
    def __init__(self, data_path, device, num_traj):
        if data_path[-4:] == '.pkl':
            raise NotImplementedError()
        else:
            # keys=['states', 'observations', 'success', 'next_observations', 'rewards', 'actions', 'truncated', 'terminated']
            keys=['observations', 'next_observations', 'rewards', 'actions', 'truncated', 'terminated']
            trajectories = load_demo_dataset(data_path, keys=keys, num_traj=num_traj, concat=False)
            print(f"trajectories have keys: {trajectories.keys()}")
            # trajectories['observations'] is a list of np.ndarray (L+1, obs_dim)
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)

        for k, v in trajectories.items():
            for i in range(len(v)):
                trajectories[k][i] = torch.Tensor(v[i]).to(device)

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,), device=device)
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        # else:
        #     raise NotImplementedError(f'Control Mode {args.control_mode} not supported')
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = args.obs_horizon, args.pred_horizon
        self.slices = []
        num_traj = len(trajectories['actions'])
        total_transitions = 0
        for traj_idx in tqdm(range(num_traj)):
            L = trajectories['actions'][traj_idx].shape[0]
            assert trajectories['observations'][traj_idx].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1 # 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon # 14
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon) for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")

        self.trajectories = trajectories
        print(f"observations: {self.trajectories['observations'][0].shape}")
        print(f"next_observations: {self.trajectories['next_observations'][0].shape}")
        print(f"reward: {self.trajectories['rewards'][0].shape}")
        print(f"actions: {self.trajectories['actions'][0].shape}")
        print(f"truncated: {self.trajectories['truncated'][0].shape}")
        print(f"terminated: {self.trajectories['terminated'][0].shape}")

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories['actions'][traj_idx].shape

        obs_seq = self.trajectories['observations'][traj_idx][max(0, start):start+self.obs_horizon]
        # start+self.obs_horizon is at least 1
        full_obs_seq = self.trajectories['observations'][traj_idx][max(0, start):end]
        full_next_obs_seq = self.trajectories['next_observations'][traj_idx][max(0, start):end]
        full_act_seq = self.trajectories['actions'][traj_idx][max(0, start):end]
        full_rewards_seq = self.trajectories['rewards'][traj_idx][max(0, start):end]
        full_truncated_seq = self.trajectories['truncated'][traj_idx][max(0, start):end]
        full_terminated_seq = self.trajectories['terminated'][traj_idx][max(0, start):end]

        if start < 0: # pad before the trajectory
            obs_seq = torch.cat([obs_seq[0].repeat(-start, 1), obs_seq], dim=0)

            full_obs_seq = torch.cat([full_obs_seq[0].repeat(-start, 1), full_obs_seq], dim=0)
            full_next_obs_seq = torch.cat([full_next_obs_seq[0].repeat(-start, 1), full_next_obs_seq], dim=0)
            full_act_seq = torch.cat([full_act_seq[0].repeat(-start, 1), full_act_seq], dim=0)

            # pad the first reward value as reward
            full_rewards_seq = torch.cat([full_rewards_seq[0].repeat(-start, ), full_rewards_seq], dim=0)
            # pad 0 as reward
            # full_rewards_seq = torch.cat([torch.zeros(-start, device=full_rewards_seq.device), full_rewards_seq], dim=0)
            
            full_truncated_seq = torch.cat([full_truncated_seq[0].repeat(-start, ), full_truncated_seq], dim=0)
            full_terminated_seq = torch.cat([full_terminated_seq[0].repeat(-start, ), full_terminated_seq], dim=0)

        if end > L: # pad after the trajectory
            # full_obs_seq.shape = (L+1, dim) -> the last step with no action
            full_obs_seq = torch.cat([full_obs_seq, full_obs_seq[-1].repeat(end-L-1, 1)], dim=0)
            full_next_obs_seq = torch.cat([full_next_obs_seq, full_next_obs_seq[-1].repeat(end-L-1, 1)], dim=0)

            gripper_action = full_act_seq[-1, -1]
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            full_act_seq = torch.cat([full_act_seq, pad_action.repeat(end-L, 1)], dim=0)

            full_rewards_seq = torch.cat([full_rewards_seq, full_rewards_seq[-1].repeat(end-L, )], dim=0)
            full_truncated_seq = torch.cat([full_truncated_seq, full_truncated_seq[-1].repeat(end-L, )], dim=0)
            full_terminated_seq = torch.cat([full_terminated_seq, full_terminated_seq[-1].repeat(end-L, )], dim=0)

            # making the robot (arm and gripper) stay still
        assert obs_seq.shape[0] == self.obs_horizon and full_act_seq.shape[0] == self.pred_horizon

        # output pred_horizon length sequence
        act_seq = full_act_seq

        return {
            'observations': obs_seq,
            'actions': act_seq,

            'full_observations': full_obs_seq, 
            'full_next_observations': full_next_obs_seq,
            'full_actions': full_act_seq,
            'full_rewards': full_rewards_seq,
            'full_truncated': full_truncated_seq,
            'full_terminated': full_terminated_seq,
        }

    def __len__(self):
        return len(self.slices)

# ALGO LOGIC: initialize Obspredictor here:
class ObsPredictor(nn.Module):
    """
    input:  state -> (B, state_dim)
            action -> (B, action_dim)

    output: next state

    In semantic, state is equivalent to obs.
    """
    def __init__(self, env):
        super(ObsPredictor, self).__init__()

        self.act_dim = env.single_action_space.shape[0]
        self.state_dim = env.single_observation_space.shape[1] # (obs_horizon, state_dim)
        hidden_dim=256

        self.shared_model = nn.Sequential(nn.Linear(self.state_dim + self.act_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish())
        # Output head 1: Predict next state
        self.obs_predictor = nn.Linear(hidden_dim, self.state_dim)
        # Output head 2: Predict reward
        self.reward_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        # Pass through the shared model
        x = self.shared_model(x)
        # Head 1: Predict next state
        next_state = self.obs_predictor(x)
        # Head 2: Predict scalar value
        next_reward = self.reward_predictor(x)
        # next_reward = torch.clamp(next_reward, min=0.0, max=1.0)
        return next_state, next_reward

# ALGO LOGIC: initialize Critic here:
class Critic(nn.Module):
    def __init__(self, env):
        super(Critic, self).__init__()

        self.act_dim = env.single_action_space.shape[0]
        self.state_dim = env.single_observation_space.shape[1] # (obs_horizon, state_dim)
        hidden_dim=256

        self.q1_model = nn.Sequential(nn.Linear(self.state_dim + self.act_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(self.state_dim + self.act_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1_model(x) 
        q2 = self.q2_model(x)
        # q1 = torch.clamp(q1, min=0.0, max=1.0)
        # q2 = torch.clamp(q2, min=0.0, max=1.0)
        return q1, q2

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1_model(x)
        # q1 = torch.clamp(q1, min=0.0, max=1.0)
        return q1

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Actor(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space.shape) == 2 # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1 # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim, # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=np.prod(env.single_observation_space.shape), # obs_horizon * obs_dim
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = args.num_diffusion_iters
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2', # has big impact on performance, try not to change
            clip_sample=True, # clip output to [-1,1] to improve stability
            prediction_type='epsilon' # predict noise (instead of denoised action)
        )

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(
            action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond)

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end] # (B, act_horizon, act_dim)

def generate_exponential_decay_sequence(N, lambd):
    """
    input: 
        N -> num of elements
        lambda -> decay rate
    x_i = e^{- \lambda i}
    """
    # generate exponential decay sequence
    sequence = np.exp(-lambd * np.arange(N))
    # sum the elements in sequence to 1
    normalized_sequence = sequence / np.sum(sequence)
    normalized_sequence = torch.Tensor(normalized_sequence)
    return normalized_sequence


def save_ckpt(run_name, tag):
    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
    ema.copy_to(ema_model.parameters())
    torch.save({
        'obsPredictor': obsPredictor.state_dict(),
        'critic': critic.state_dict(),
        'critic_target': critic_target.state_dict(),
        'actor': actor.state_dict(),
        'ema_model': ema_model.state_dict(),
    }, f'runs/{run_name}/checkpoints/{tag}.pt')



if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith('.h5'):
        import json
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state", render_mode="rgb_array")
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, other_kwargs, video_dir=f'runs/{run_name}/videos' if args.capture_video else None)

    if args.track:
        import wandb
        config = vars(args)
        # config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=gym_utils.find_max_episode_steps_value(envs))
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy",
            tags=["diffusion_policy"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # dataloader setup
    dataset = SmallDemoDataset_DiffusionPolicy(args.demo_path, device, num_traj=args.num_demos)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )
    if args.num_demos is None:
        args.num_demos = len(dataset)

    # actor setup
    actor = Actor(envs, args).to(device)
    actor_optimizer = optim.AdamW(params=actor.parameters(),
        lr=args.actor_lr, betas=(0.95, 0.999), weight_decay=1e-6)
    
    critic = Critic(envs).to(device)
    critic_target = Critic(envs).to(device)
    critic_target = copy.deepcopy(critic)
    critic_optimizer = optim.Adam(list(critic.parameters()), lr=args.critic_lr)

    obsPredictor = ObsPredictor(envs).to(device)
    obsPredictor_optimizer = optim.Adam(list(obsPredictor.parameters()), lr=args.obsPredictor_lr)

    # Cosine LR schedule with linear warmup
    critic_lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=critic_optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )
    
    actor_lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=actor_optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=actor.parameters(), power=0.75)
    ema_model = Actor(envs, args).to(device)

    # ---------------------------------------------------------------------------- #
    # Other hyperparameters
    # ---------------------------------------------------------------------------- #
    horizon_comp_mode = 'exponential_decay' # uniform or exponential_decay
    if horizon_comp_mode == 'exponential_decay':
        critic_horizon_comp_factor = generate_exponential_decay_sequence(args.act_horizon-1, 0.25).to(device)
        actor_horizon_comp_factor = generate_exponential_decay_sequence(args.act_horizon, 0.25).to(device)


    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    actor.train()

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    for iteration, data_batch in enumerate(train_dataloader):

        # read data from batch
        full_observations = data_batch['full_observations']             # (B, pred_horizon, obs_dim)
        full_next_observations = data_batch['full_next_observations']   # (B, pred_horizon, obs_dim)
        full_actions = data_batch['full_actions']                       # (B, pred_horizon, act_dim)
        full_rewards = data_batch['full_rewards']                       # (B, pred_horizon)
        full_truncated = data_batch['full_truncated']                   # (B, pred_horizon)
        full_terminated = data_batch['full_terminated']                 # (B, pred_horizon)

        dones = (full_truncated.int() | full_terminated.int()).float()  # (B, pred_horizon)

        # # copy data from cpu to gpu
        # data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        ################################
        ###### ObsPredictor Train ######
        ################################
        obsPredictor_starttime = datetime.datetime.now()

        obsPredictor_obs_loss = 0.0
        obsPredictor_reward_loss = 0.0
        obsPredictor_loss = 0.0
        for t in range(args.pred_horizon):
            obs_t = full_observations[:, t, :]
            act_t = full_actions[:, t, :]
            true_obs_tp1 = full_next_observations[:, t, :]
            true_reward_t = full_rewards[:, t]

            pred_obs_tp1, pred_reward_t = obsPredictor.forward(obs_t, act_t)

            obsPredictor_obs_loss += F.mse_loss(pred_obs_tp1, true_obs_tp1)
            obsPredictor_reward_loss += F.mse_loss(pred_reward_t, true_reward_t.unsqueeze(-1))
        obsPredictor_loss = obsPredictor_obs_loss + obsPredictor_reward_loss
            # obsPredictor_loss = F.mse_loss(pred_obs_tp1, true_obs_tp1) + F.mse_loss(pred_reward_t, true_reward_t.unsqueeze(-1))

        obsPredictor_optimizer.zero_grad()
        obsPredictor_loss.backward()
        obsPredictor_optimizer.step()

        obsPredictor_endtime = datetime.datetime.now()

        ################################
        ######### Critic Train #########
        ################################
        critic_starttime = datetime.datetime.now()

        with torch.no_grad():            
            # shape (B, act_horizon-1)
            not_done = 1 - dones[:, args.obs_horizon:args.obs_horizon+args.act_horizon-1]
            # truncate shape to (B, act_horizon-1, dim)
            if iteration < args.critic_warmup_steps:
                # shape (B, act_horizon-1)
                cumulative_rewards = full_rewards[:, args.obs_horizon:args.obs_horizon+args.act_horizon-1]
                next_obs = full_observations[:, args.obs_horizon:args.obs_horizon+args.act_horizon-1]
                next_obs_action = full_actions[:, args.obs_horizon:args.obs_horizon+args.act_horizon-1]
            else:
                if iteration == args.critic_warmup_steps:
                    print(f"Critic using Actor predicting action sequence strategy start from iteration {iteration}...")
                next_obs = []
                cumulative_rewards = []
                next_obs_action = ema_model.get_action(data_batch['observations'])
                obs_t = data_batch['observations'][:, -1]
                for t in range(args.act_horizon-1):
                    pred_obs_tp1, pred_reward_t = obsPredictor.forward(obs_t, next_obs_action[:, t])
                    obs_t = pred_obs_tp1
                    next_obs.append(obs_t)
                    cumulative_rewards.append(pred_reward_t.squeeze(-1))
                next_obs = torch.stack(next_obs, dim=1)
                cumulative_rewards = torch.stack(cumulative_rewards, dim=-1)
            
            # target_Q require:
            # 1. not_done -> (B, act_horizon-1)
            # 2. cumulative_rewards -> (B, act_horizon-1)   
            # 3. next_obs -> (B, act_horizon-1, obs_dim)
            # 4. next_obs_action -> (B, act_horizon-1, act_dim)
            target_Q = []
            for t in range(args.act_horizon-1):
                min_target_q_t = critic_target.q_min(next_obs[:, t], next_obs_action[:, t]) # (B, 1)
                target_q_t = cumulative_rewards[:, t] + not_done[:, t] * args.gamma * min_target_q_t.squeeze(-1)
                # target_q_t = torch.where(not_done[:, t] == 0, torch.ones_like(target_q_t), target_q_t)
                target_Q.append(target_q_t)
            target_Q = torch.stack(target_Q, dim=1) # (B, args.act_horizon-1)

            critic_endtime = datetime.datetime.now()

        # (B, 1)
        current_Q1, current_Q2 = critic(full_observations[:, args.obs_horizon-1], full_actions[:, args.obs_horizon-1])
        if horizon_comp_mode == 'uniform':
            target_Q_mean = torch.mean(target_Q, dim=1, keepdim=True) # (B, 1)
        elif horizon_comp_mode == 'exponential_decay':
            expanded_critic_horizon_comp_factor = critic_horizon_comp_factor.expand(target_Q.shape[0], -1)
            weighted_target_Q = expanded_critic_horizon_comp_factor * target_Q
            target_Q_mean = weighted_target_Q.sum(dim=1, keepdim=True) 
        else:
            raise RuntimeError("This type of horizon computation mode is not defined.")
        

        critic_loss = F.mse_loss(current_Q1, target_Q_mean) + F.mse_loss(current_Q2, target_Q_mean)
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        if args.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.grad_norm, norm_type=2)
        critic_optimizer.step()
        critic_lr_scheduler.step()
        
        
        ################################
        ######### Policy Train #########
        ################################
        actor_starttime = datetime.datetime.now()

        # Actor forward and compute loss
        bc_loss = actor.compute_loss(
            obs_seq=data_batch['observations'], # (B, obs_horizon, obs_dim) # especially designed for actor input
            action_seq=data_batch['actions'], # (B, pred_horizon, act_dim)
        )

        act_seq = actor.get_action(data_batch['observations']) # (B, act_horizon, act_dim)
        Q_seq = []
        obs_t = data_batch['observations'][:, -1]
        act_t = act_seq[:, 0]
        for t in range(args.act_horizon):
            Q = critic.q1(obs_t, act_t).squeeze(-1)
            obs_t, _ = obsPredictor(obs_t, act_t)
            act_t = act_seq[:, t]
            Q_seq.append(Q)
        Q_seq = torch.stack(Q_seq, dim=-1)

        if horizon_comp_mode == 'uniform':
            Q_seq_mean = Q_seq.mean() # (1,)
        elif horizon_comp_mode == 'exponential_decay':
            expanded_actor_horizon_comp_factor = actor_horizon_comp_factor.expand(Q_seq.shape[0], -1)
            weighted_Q_seq = expanded_actor_horizon_comp_factor * Q_seq
            Q_seq_mean = weighted_Q_seq.sum(dim=1, keepdim=True) 
        else:
            raise RuntimeError("This type of horizon computation mode is not defined.")

        lmbda = args.alpha/(Q_seq.abs().mean().detach())
        q_loss = -lmbda * Q_seq_mean.mean() 
        # actor_loss = q_loss + args.beta * bc_loss
        actor_loss = q_loss + bc_loss

        actor_optimizer.zero_grad()
        actor_loss.backward()
        # if args.grad_norm > 0:
        #     actor_grad_norms = nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.grad_norm, norm_type=2)
        actor_optimizer.step()
        actor_lr_scheduler.step() # step lr scheduler every batch, this is different from standard pytorch behavior

        last_tick = time.time()

        actor_endtime = datetime.datetime.now()

        ################################
        #### Update Auxiliary Model ####
        ################################
        # update Exponential Moving Average of the model weights
        ema.step(actor.parameters())
        # update Target Critic
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        ################################
        ########### Log Data ###########
        ################################        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if iteration % args.log_freq == 0:
            obsPredictor_runtime = (obsPredictor_endtime - obsPredictor_starttime).total_seconds() / 60
            critic_runtime = (critic_endtime - critic_starttime).total_seconds() / 60
            actor_runtime = (actor_endtime - actor_starttime).total_seconds() / 60
            print(f"\n################### Iteration {iteration}, Progress {iteration/args.total_iters*100}% ########################")
            print(f"------------------------------------------------------")
            print(f"obsPredictor_runtime: {obsPredictor_runtime * args.log_freq} mins.")
            print(f"critic_runtime: {critic_runtime * args.log_freq} mins.")
            print(f"actor_runtime: {actor_runtime * args.log_freq} mins.")
            print(f"------------------------------------------------------")
            print(f"obsPredictor_obs_loss: {obsPredictor_obs_loss.item()}")
            print(f"obsPredictor_reward_loss: {obsPredictor_reward_loss.item()}")
            print(f"obsPredictor_loss: {obsPredictor_loss.item()}")
            print(f"------------------------------------------------------")
            print(f"Q1: {current_Q1.mean().item()}, Q2: {current_Q2.mean().item()}, target_Q_mean[:4, :]: {target_Q_mean[:4, :].flatten()}")
            # print(f"current_Q1: {current_Q1[:4, :].flatten()}")
            # print(f"current_Q2: {current_Q2[:4, :].flatten()}")
            # print(f"target_Q: {target_Q[:4, :]}")
            # print(f"actor_Q_seq: {Q_seq[:4, :]}")
            print(f"critic_loss: {critic_loss.item()}")
            print(f"------------------------------------------------------")
            print(f"q_loss: {q_loss.item()}, Q: {Q_seq.mean().item()}")
            print(f"bc_loss: {bc_loss.item()}")
            print(f"actor_loss: {actor_loss.item()}")
            print(f"------------------------------------------------------")
            print(f"################### Iteration {iteration}, Progress {iteration/args.total_iters*100}% ########################\n")
            
            writer.add_scalar("charts/actor_learning_rate", actor_optimizer.param_groups[0]["lr"], iteration)
            writer.add_scalar("charts/critic_learning_rate", critic_optimizer.param_groups[0]["lr"], iteration)
            # writer.add_scalar("charts/actor_grad_norms", actor_grad_norms.max().item(), iteration)
            writer.add_scalar("charts/critic_grad_norms", critic_grad_norms.max().item(), iteration)

            writer.add_scalar("losses/obsPredictor_obs_loss", obsPredictor_obs_loss.item(), iteration)
            writer.add_scalar("losses/obsPredictor_reward_loss", obsPredictor_reward_loss.item(), iteration)
            writer.add_scalar("losses/obsPredictor_loss", obsPredictor_loss.item(), iteration)
            
            writer.add_scalar("losses/current_Q1", current_Q1.mean().item(), iteration)
            writer.add_scalar("losses/current_Q2", current_Q2.mean().item(), iteration)
            writer.add_scalar("losses/target_Q_mean", target_Q_mean.mean().item(), iteration)
            writer.add_scalar("losses/critic_loss", critic_loss.item(), iteration)
            
            writer.add_scalar("losses/q_loss", q_loss.item(), iteration)
            writer.add_scalar("losses/Q_value", Q_seq.mean().item(), iteration)
            writer.add_scalar("losses/bc_loss", bc_loss.item(), iteration)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)
        # Evaluation
        if iteration % args.eval_freq == 0:
            last_tick = time.time()
            eval_starttime = datetime.datetime.now()

            ema.copy_to(ema_model.parameters())
            # def sample_fn(obs):

            eval_metrics = evaluate(args.num_eval_episodes, ema_model, envs, device, args.sim_backend)
            timings["eval"] += time.time() - last_tick
            eval_endtime = datetime.datetime.now()
            eval_runtime = (eval_endtime - eval_starttime).total_seconds() / 60
            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            print(f"eval_runtime: {eval_runtime} minutes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(f'New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint.')
        # Checkpoint
        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration))
    envs.close()
    writer.close()
