import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchMultiCategorical

from utils import ActionAdapter

import logging
import logging_setup


class ImpalaMlpModule(TorchRLModule, ValueFunctionAPI):
    """
    RLModule to use with IMPALA and have full control of the network parameters.
    """

    # 1) Build the sub-nets here
    def setup(self):

        self.logger = logging.getLogger("MyRLApp.RLModule")

        self.logger.info("MLPActorCriticModule.setup()")

        self.action_adapter = ActionAdapter(self.action_space)
        self.lens = tuple([int(k) for k in self.action_adapter.nvec])
        self.logger.debug(f"lens: {self.lens}")

        self._build_networks()

    # 2) One shared forward – used by exploration, train, inference
    def _forward(self, batch, **kwargs):
        obs = batch[Columns.OBS].float()

        # actor
        encoding = self.actor_encoder(obs)
        if self.cont:
            mu = self.mu_head(encoding)
            log_std = self.log_std.expand_as(mu)
            action_inputs = torch.cat([mu, log_std], dim=-1)
        else:
            action_inputs = self.logits_head(encoding)

        return {
            Columns.ACTION_DIST_INPUTS: action_inputs,
        }

    def _forward_train(self, batch, **kwargs):
        obs = batch[Columns.OBS].float()

        # critic
        values = self.value_head(self.actor_encoder(obs)).squeeze(-1)

        # actor
        encoding = self.actor_encoder(obs)
        if self.cont:
            mu = self.mu_head(encoding)
            log_std = self.log_std.expand_as(mu)
            action_inputs = torch.cat([mu, log_std], dim=-1)
        else:
            action_inputs = self.logits_head(encoding)

        return {
            Columns.ACTION_DIST_INPUTS: action_inputs,
            Columns.EMBEDDINGS: values,     # this will get passed down to compute_values()
        }

    def compute_values(self, batch, embeddings=None):
        # Re-use cached critic output if caller provides it
        if embeddings is None:
            obs = batch[Columns.OBS].float()
            return self.value_head(self.actor_encoder(obs)).squeeze(-1)

        assert isinstance(embeddings, torch.Tensor)
        return embeddings

    def _build_networks(self):
        self.logger.debug("In MLPActorCriticModule._build_networks()")

        if hasattr(self, "actor_encoder"):
            return

        self.logger.debug("Does not have actor_encoder, building it.")

        if isinstance(self.observation_space, gym.spaces.Box):
            obs_dim = int(np.prod(self.observation_space.shape))
        elif isinstance(self.observation_space, gym.spaces.Discrete):
            obs_dim = 1
        else:
            raise NotImplementedError(f"Unsupported obs space {self.observation_space}")

        # extract action space information and make actor head

        if self.action_adapter.mode in ("discrete1", "multidiscrete"):
            # discrete action space case
            self.cont = False
            act_dim = self.action_adapter.nint
            self.logits_head = nn.Linear(64, act_dim)
        elif self.action_adapter.mode == "continuous":
            # continuous action space case
            self.cont = True
            act_dim = self.action_space.shape[0]
            self.mu_head = nn.Linear(64, act_dim)
            # for the log head, this can be state-dependent or not. SAC uses state-dependent, default IMPALA uses
            # state-independent. to make it state-dependent, the log head would have to be a nn.Linear(XX, act_dim)
            # and its output needs to be clipped, something like torch.clamp(self.log_std_head(feat), min=-20., max=2.),
            # to keep the std from being too small or too large.
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        else:
            raise NotImplementedError(f"Unsupported space {self.action_space}")

        activation = nn.SiLU()

        # -------- actor ----------
        self.actor_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), activation,
            nn.Linear(256, 256), activation,
            nn.Linear(256, 64), activation,
        )

        # -------- critic ----------
        self.critic_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), activation,
            nn.Linear(64, 64), activation,
            nn.Linear(64, 32), activation,
        )
        self.value_head = nn.Linear(64, 1)

        self.logger.debug(f"actor encoder: {self.actor_encoder}")
        self.logger.debug(f"actor head: {self.logits_head}")

    def get_inference_action_dist_cls(self):
        if isinstance(self.action_space, gym.spaces.Tuple):
            return self._tuple_multi_cat_cls(self.lens)
        return super().get_inference_action_dist_cls()

    def get_exploration_action_dist_cls(self):
        if isinstance(self.action_space, gym.spaces.Tuple):
            return self._tuple_multi_cat_cls(self.lens)
        return super().get_exploration_action_dist_cls()

    def get_train_action_dist_cls(self):
        if isinstance(self.action_space, gym.spaces.Tuple):
            return self._tuple_multi_cat_cls(self.lens)
        return super().get_train_action_dist_cls()

    def _tuple_multi_cat_cls(self, lens):

        class FixedMultiCategorical(TorchMultiCategorical):
            @classmethod
            def from_logits(cls, logits):
                return super().from_logits(logits, lens)
        return FixedMultiCategorical

