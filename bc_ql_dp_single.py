from collections import defaultdict
import os
import random
from dataclasses import dataclass
import time
from typing import Optional
import copy

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro
import wandb
from mani_skill.utils import gym_utils
from mani_skill.utils.io_utils import load_json
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from tqdm import tqdm

from behavior_cloning.make_env import make_eval_envs
from behavior_cloning.evaluate import evaluate

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from behavior_cloning.conditional_unet1d import MLP
from dataclasses import dataclass, field
from typing import Optional, List



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
    """whether to capture videos of the agent performances (check out `videos` folder)"""

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
    # num_demos: Optional[int] = 10
    # """number of trajectories to load from the demo dataset"""
    # total_iters: int = 10000
    # """total timesteps of the experiment"""
    # batch_size: int = 4
    # """the batch size of sample from the replay memory"""

    # Behavior cloning specific arguments
    actor_lr: float = 3e-4
    """the learning rate for the actor"""
    critic_lr: float = 3e-4
    """the learning rate for the critic"""
    normalize_states: bool = False
    """if toggled, states are normalized to mean 0 and standard deviation 1"""
    diffusion_step_embed_dim: int = 64 # not very important
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
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 100
    """the frequency of logging the training metrics"""
    eval_freq: int = 1000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = 'pd_joint_delta_pos'
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None

# taken from here
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkillDataset(Dataset):
    """
    item return: {obs, next_obs, action, reward, done}
    
    * done: bool
    """
    def __init__(
        self,
        dataset_file: str,
        device,
        load_count=-1,
        normalize_states=False,
    ) -> None:
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]

        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.observations = []
        self.next_observations = []
        self.actions = []
        self.dones = []
        self.rewards = []
        self.total_frames = 0
        self.device = device
        if load_count is None:
            load_count = len(self.episodes)
        print(f"Loading {load_count} episodes")

        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            # trajectory["obs"] -> T+1 * D
            # trajectory["actions"] -> T * A
            self.observations.append(trajectory["obs"][:-1])
            self.next_observations.append(trajectory["obs"][1:])
            self.actions.append(trajectory["actions"])
            self.rewards.append(trajectory["rewards"].reshape(-1, 1))
            self.dones.append(trajectory["success"].reshape(-1, 1))

        self.observations = np.vstack(self.observations)
        self.next_observations = np.vstack(self.next_observations)
        self.actions = np.vstack(self.actions)
        self.rewards = np.vstack(self.rewards)
        self.dones = np.vstack(self.dones)
        
        assert self.observations.shape[0] == self.actions.shape[0]
        assert self.next_observations.shape[0] == self.actions.shape[0]
        assert self.dones.shape[0] == self.actions.shape[0]
        assert self.rewards.shape[0] == self.actions.shape[0]

        if normalize_states:
            mean, std = self.get_state_stats()
            self.observations = (self.observations - mean) / std

    def get_state_stats(self):
        return np.mean(self.observations), np.std(self.observations)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        action = torch.from_numpy(self.actions[idx]).float().to(device=self.device)
        obs = torch.from_numpy(self.observations[idx]).float().to(device=self.device)
        next_obs = torch.from_numpy(self.next_observations[idx]).float().to(device=self.device)
        reward = torch.from_numpy(self.rewards[idx]).float().to(device=self.device)
        done = torch.from_numpy(self.dones[idx]).to(device=self.device)
        return obs, next_obs, action, reward, done

# ALGO LOGIC: initialize Critic here:
class Critic(nn.Module):
    def __init__(self, env):
        super(Critic, self).__init__()

        self.obs_dim = envs.single_observation_space.shape[0] 
        self.act_dim = envs.single_action_space.shape[0]
        hidden_dim=256

        self.q1_model = nn.Sequential(nn.Linear(self.obs_dim + self.act_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(self.obs_dim + self.act_dim, hidden_dim),
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

        
        assert len(env.single_observation_space.shape) == 1 # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1 # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.obs_dim = env.single_observation_space.shape[0] 
        self.act_dim = env.single_action_space.shape[0]

        self.noise_pred_net = MLP(
            action_dim=self.act_dim, # act_horizon is not used (U-Net doesn't care)
            state_dim=self.obs_dim, # obs_horizon * obs_dim
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        )
        self.num_diffusion_iters = 5 #100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2', # has big impact on performance, try not to change
            clip_sample=True, # clip output to [-1,1] to improve stability
            prediction_type='epsilon' # predict noise (instead of denoised action)
        )

    def compute_loss(self, obs, action):
        """
        obs: (B, dim), e.g. (batch_size, 42)
        action: (B, dim), e.g. (batch_size, 4)
        """
        B = obs.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs.flatten(start_dim=1) # adapted from (B, obs_horizon * obs_dim) but does not matter here

        # sample noise to add to actions
        noise = torch.randn((B, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs.shape[0]
        with torch.no_grad():
            obs_cond = obs.flatten(start_dim=1) # adapted from (B, obs_horizon * obs_dim) but does not matter here

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn((B, self.act_dim), device=obs.device)

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

        return noisy_action_seq # (B, act_dim)


def save_ckpt(run_name, tag):
    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
    torch.save({
        "actor": actor.state_dict(),
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

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state", render_mode="rgb_array")
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, video_dir=f'runs/{run_name}/videos' if args.capture_video else None)

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
            group="BehaviorCloning",
            tags=["behavior_cloning"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    ds = ManiSkillDataset(
        args.demo_path,
        device=device,
        load_count=args.num_demos,
        normalize_states=args.normalize_states,
    )

    obs, _ = envs.reset(seed=args.seed)

    sampler = RandomSampler(ds)
    batchsampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    itersampler = IterationBasedBatchSampler(batchsampler, args.total_iters)
    dataloader = DataLoader(ds, batch_sampler=itersampler, num_workers=args.num_dataload_workers)

    actor = Actor(env=envs, args=args).to(device=device)
    # optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    actor_optimizer = optim.AdamW(params=actor.parameters(),
        lr=args.actor_lr, betas=(0.95, 0.999), weight_decay=1e-6)
    # Cosine LR schedule with linear warmup
    actor_lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=actor_optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    critic = Critic(envs).to(device)
    critic_target = Critic(envs).to(device)
    critic_target = copy.deepcopy(critic)
    critic_optimizer = optim.AdamW(params=critic.parameters(),
        lr=args.critic_lr, betas=(0.95, 0.999), weight_decay=1e-6)
    critic_lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=critic_optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=actor.parameters(), power=0.75)
    ema_agent = Actor(envs, args).to(device)

    best_eval_metrics = defaultdict(float)


    for iteration, batch in enumerate(dataloader):
        log_dict = {}
        obs, next_obs, action, reward, done = batch 
        
        ################################
        ######### Critic Train #########
        ################################
        with torch.no_grad():
            next_action = ema_agent.get_action(next_obs)
            target_Q = critic_target.q_min(next_obs, next_action)
            target_Q = reward + (1 - done.float()) * args.gamma * target_Q
            
        
        current_Q1, current_Q2 = critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        if args.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.grad_norm, norm_type=2)
        critic_optimizer.step()
        critic_lr_scheduler.step()

        ################################
        ######### Policy Train #########
        ################################
        bc_loss = actor.compute_loss(obs, action)
        action_pred = actor.get_action(obs)
        Q = critic.q1(obs, action_pred)
        lmbda = args.alpha/(Q.abs().mean().detach())
        q_loss = -lmbda * Q.mean()
        actor_loss = q_loss + bc_loss + F.mse_loss(action_pred, action) 
        # actor_loss = q_loss + bc_loss

        actor_optimizer.zero_grad()
        actor_loss.backward()
        # if args.grad_norm > 0:
        #     actor_grad_norms = nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.grad_norm, norm_type=2)
        actor_optimizer.step()
        actor_lr_scheduler.step() # step lr scheduler every batch, this is different from standard pytorch behavior
        
        ################################
        #### Update Auxiliary Model ####
        ################################
        # update Exponential Moving Average of the model weights
        ema.step(actor.parameters())
        # update Target Critic
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)



        if iteration % args.log_freq == 0:
            print(f"Iteration {iteration}, critic_loss: {critic_loss.item():.6f}, actor_loss: {actor_loss:.6f} = {bc_loss.item():.6f} + {q_loss.item():.6f}")
            writer.add_scalar("charts/actor_learning_rate", actor_optimizer.param_groups[0]["lr"], iteration)
            writer.add_scalar("charts/critic_learning_rate", critic_optimizer.param_groups[0]["lr"], iteration)
            # writer.add_scalar("charts/actor_grad_norms", actor_grad_norms.max().item(), iteration)
            writer.add_scalar("charts/critic_grad_norms", critic_grad_norms.max().item(), iteration)

            writer.add_scalar("losses/critic_loss", critic_loss.item(), iteration)
            writer.add_scalar("losses/bc_loss", bc_loss.item(), iteration)
            writer.add_scalar("losses/q_loss", q_loss.item(), iteration)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), iteration)

        if iteration % args.eval_freq == 0:
            ema.copy_to(ema_agent.parameters())
            actor.eval()
            ema_agent.eval()
            def sample_fn(obs):
                if isinstance(obs, np.ndarray):
                    obs = torch.from_numpy(obs).float().to(device)
                action = ema_agent.get_action(obs)
                if args.sim_backend == "cpu":
                    action = action.cpu().numpy()
                return action
            with torch.no_grad():
                eval_metrics = evaluate(args.num_eval_episodes, sample_fn, envs)
            actor.train()
            ema_agent.train()
            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
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

        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration))
    envs.close()
    wandb.finish()
