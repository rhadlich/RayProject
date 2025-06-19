import torch
import torch.nn as nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule

from utils import ActionAdapter


class MLPActorCriticModule(TorchRLModule, ValueFunctionAPI):
    """
    Actor: encoder [128, 128] + post layer [64]  (ReLU)
    Critic: encoder [64, 64]  + post layer [32]  (ReLU)
    """

    # 1) Build the sub-nets here
    def setup(self):
        obs_dim = self.observation_space.shape[0]

        # extract action space information and make actor head
        action_adapter = ActionAdapter(self.action_space)
        if action_adapter.mode in ("discrete1", "multidiscrete"):
            # discrete action space case
            self.cont = False
            act_dim = self.nint
            self.logits_head = nn.Linear(64, act_dim)
        elif action_adapter.mode == "continuous":
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

        # -------- actor ----------
        self.actor_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )

        # -------- critic ----------
        self.critic_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.value_head = nn.Linear(32, 1)

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
        values = self.value_head(self.critic_encoder(obs)).squeeze(-1)

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
            return self.value_head(self.critic_encoder(obs)).squeeze(-1)

        assert isinstance(embeddings, torch.Tensor)
        return embeddings
