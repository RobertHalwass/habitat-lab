import torch
import numpy as np
import random

from gym.spaces import Dict as SpaceDict
from gym.spaces import Box
from gym.spaces import Discrete

from habitat import Config
from habitat.core.spaces import EmptySpace
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.utils.common import (
    batch_obs,
    ObservationBatchingCache
)
from habitat_baselines.config.default import get_config
from torch import nn       

import torchvision.transforms as transforms
import time


class PointNavAgent:
    def __init__(self, model_path: str, config_path, input_type: str = "rgbd") -> None:
        ckpt_dict = torch.load(model_path, map_location="cpu")
        start_config = get_config(config_path)
        self.config = self.setup_eval_config(ckpt_dict["config"], start_config)

        random.seed(start_config.TASK_CONFIG.SEED)
        np.random.seed(start_config.TASK_CONFIG.SEED)
        torch.manual_seed(start_config.TASK_CONFIG.SEED)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.ppo_cfg = self.config.RL.PPO

        self.input_type = input_type

        self._obs_batching_cache = ObservationBatchingCache()

        self.policy_action_space = Discrete(4)
        self.action_shape = (1,)

        self.setup_actor_critic_agent()

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        self.actor_critic.eval()

        self.test_recurrent_hidden_states = torch.zeros(
            (1),
            self.actor_critic.net.num_recurrent_layers,
            self.ppo_cfg.hidden_size,
            device=self.device,
        )
        self.prev_actions = torch.zeros(
            (1),
            (1),
            device=self.device,
            dtype=torch.long,
        )
        self.not_done_masks = torch.zeros(
            (1),
            (1),
            device=self.device,
            dtype=torch.bool,
        )

    def act(self, observations):
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        with torch.no_grad():
            _, actions, _, self.test_recurrent_hidden_states = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            self.prev_actions.copy_(actions)  # type: ignore

        self.not_done_masks = torch.tensor(
            [True],
            dtype=torch.bool,
            device=self.device,
        )
        return actions[0].item()

    def reset(self):
        self.test_recurrent_hidden_states = torch.zeros(
            (1),
            self.actor_critic.net.num_recurrent_layers,
            self.ppo_cfg.hidden_size,
            device=self.device,
        )
        self.prev_actions = torch.zeros(
            (1),
            (1),
            device=self.device,
            dtype=torch.long,
        )
        self.not_done_masks = torch.zeros(
            (1),
            (1),
            device=self.device,
            dtype=torch.bool,
        )

    def setup_actor_critic_agent(self):
        spaces = {
            "pointgoal_with_gps_compass": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )
        }

        if self.input_type in ["depth", "rgbd"]:
            spaces["depth"] = Box(
                low=0,
                high=1,
                shape=(300, 400, 1),
                dtype=np.float32,
            )

        if self.input_type in ["rgb", "rgbd"]:
            spaces["rgb"] = Box(
                low=0,
                high=255,
                shape=(300, 400, 3),
                dtype=np.uint8,
            )
        observation_space = SpaceDict(spaces)

        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.actor_critic = PointNavResNetPolicy.from_config(
            self.config, observation_space, self.policy_action_space
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
        nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=self.ppo_cfg.clip_param,
            ppo_epoch=self.ppo_cfg.ppo_epoch,
            num_mini_batch=self.ppo_cfg.num_mini_batch,
            value_loss_coef=self.ppo_cfg.value_loss_coef,
            entropy_coef=self.ppo_cfg.entropy_coef,
            lr=self.ppo_cfg.lr,
            eps=self.ppo_cfg.eps,
            max_grad_norm=self.ppo_cfg.max_grad_norm,
            use_normalized_advantage=self.ppo_cfg.use_normalized_advantage,
        )

    def setup_eval_config(self, checkpoint_config: Config, start_config: Config) -> Config:
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        config = start_config.clone()

        ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        try:
            config.merge_from_other_cfg(checkpoint_config)
            config.merge_from_other_cfg(start_config)
            config.merge_from_list(ckpt_cmd_opts)
            config.merge_from_list(eval_cmd_opts)
        except KeyError:
            logger.info("Saved config is outdated, using solely eval config")
            config = self.config.clone()
        config.defrost()
        if config.TASK_CONFIG.DATASET.SPLIT == "train":
            config.TASK_CONFIG.DATASET.SPLIT = "val"
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = start_config.SENSORS
        config.freeze()

        return config
