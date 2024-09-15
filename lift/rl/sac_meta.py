from typing import Dict, Tuple
import torch
from tensordict import TensorDictBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.sac import SACLoss

from lift.rl.sac import SAC


class MetaSAC(SAC):
    """Meta learning SAC"""
    def __init__(self, config, train_env, eval_env):
        super().__init__(config, train_env, eval_env)
    
    def _init_loss_module(self):
        # Create SAC loss
        self.loss_module = MetaSACLoss(
            actor_network=self.model["policy"],
            qvalue_network=self.model["value"],
            num_qvalue_nets=2,
            loss_function=self.config.loss_function,
            delay_actor=False,
            delay_qvalue=True,
            alpha_init=self.config.alpha_init,
        )


class MetaSACLoss(SACLoss):
    """SAC loss computed based on meta parameters"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self._version == 2, "torchrl SACLoss._version must be 2"

    def _actor_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with set_exploration_type(
            ExplorationType.RANDOM
        ), self.actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(tensordict)
            a_reparm = dist.rsample()
        log_prob = dist.log_prob(a_reparm)

        td_q = tensordict.select(*self.qvalue_network.in_keys)
        td_q.set(self.tensor_keys.action, a_reparm)
        td_q = self._vmap_qnetworkN0(
            td_q,
            self._cached_detached_qvalue_params,  # should we clone?
        )
        min_q_logprob = (
            td_q.get(self.tensor_keys.state_action_value).min(0)[0].squeeze(-1)
        )

        if log_prob.shape != min_q_logprob.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_q_logprob.shape}"
            )
        
        alpha = self._alpha
        return alpha * log_prob - min_q_logprob, {"log_prob": log_prob.detach()}

    def _compute_target_v2(self, tensordict) -> torch.Tensor:
        r"""Value network for SAC v2.

        SAC v2 is based on a value estimate of the form:

        .. math::

          V = Q(s,a) - \alpha * \log p(a | s)

        This class computes this value given the actor and qvalue network

        """
        tensordict = tensordict.clone(False)
        # get actions and log-probs
        with torch.no_grad():
            with set_exploration_type(
                ExplorationType.RANDOM
            ), self.actor_network_params.to_module(self.actor_network):
                next_tensordict = tensordict.get("next").clone(False)
                next_dist = self.actor_network.get_dist(next_tensordict)
                next_action = next_dist.rsample()
                next_tensordict.set(self.tensor_keys.action, next_action)
                next_sample_log_prob = next_dist.log_prob(next_action)

            # get q-values
            next_tensordict_expand = self._vmap_qnetworkN0(
                next_tensordict, self.target_qvalue_network_params
            )
            state_action_value = next_tensordict_expand.get(
                self.tensor_keys.state_action_value
            )
            if (
                state_action_value.shape[-len(next_sample_log_prob.shape) :]
                != next_sample_log_prob.shape
            ):
                next_sample_log_prob = next_sample_log_prob.unsqueeze(-1)
            
            alpha = self._alpha
            next_state_value = state_action_value - alpha * next_sample_log_prob
            next_state_value = next_state_value.min(0)[0]
            tensordict.set(
                ("next", self.value_estimator.tensor_keys.value), next_state_value
            )
            target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)
            return target_value

    def _alpha_loss(self, log_prob: torch.Tensor) -> torch.Tensor:
        """Dummpy alpha loss multiplied with zero"""
        alpha_loss = -self.log_alpha * (log_prob + self.target_entropy)
        return alpha_loss