import copy
from tqdm import tqdm

import gymnasium as gym
import numpy as np
from gymnasium.utils.step_api_compatibility import step_api_compatibility

from lift.simulator.simulator import WindowSimulator
from lift.utils import obs_wrapper


def rollout(env, model, n_steps=1000, is_sb3=False, random_pertube_prob=0.0, action_noise=0.0):
    """
    Args:
        env: gym environment
        policy: policy class with sample_action method or predict method if stable_baseline3 policy.
        n_steps: number of steps.
        is_sb3: whether the policy is stable_baseline3 policy.
        save_data: whether to save trajectory data. If no, only save reward and other fields will be empty.
        random_pertube_prob: probability of random pertubation of teacher actions. If pertub, then random action in range [-1, 1] is taken.
        action_noise: noise added to teacher actions.

    Returns:
        data: dict with fields [obs, act, rwd, next_obs, done]. dict observations will be concatenated for each key.
    """
    data = {"obs": [], "act": [], "rwd": [], "next_obs": [], "done": []}

    observation = obs_wrapper(env.reset())
    
    bar = tqdm(range(n_steps), desc="Rollout", unit="item")
    while len(data["rwd"]) < n_steps:
        is_pertub = False
        if is_sb3:
            action, _ = model.predict(observation)

            # randomely pertube teacher actions to obtain broader diversity in data
            if np.random.rand() < random_pertube_prob:
                action = np.random.rand(*action.shape) * 2 - 1
                is_pertub = True
        else:
            action = model.sample_action(observation)
        
        action += np.random.randn(*action.shape) * action_noise

        next_observation, reward, terminated, info = step_api_compatibility(
            env.step(action), 
            output_truncation_bool=False
        )

        if not is_pertub:
            data["rwd"].append(float(reward))
            data["obs"].append(copy.deepcopy(observation))
            data["act"].append(action)
            data["next_obs"].append(copy.deepcopy(next_observation))
            data["done"].append(terminated)
            bar.update(1)
        
        observation = next_observation

        if terminated:
            observation = obs_wrapper(env.reset())
    
    if isinstance(data["obs"][0], dict):
        keys = list(data["obs"][0].keys())
        data["obs"] = {k: np.concatenate([o[k] for o in data["obs"]], axis=0) for k in keys}
        data["next_obs"] = {k: np.concatenate([o[k] for o in data["next_obs"]], axis=0) for k in keys}
    else:
        data["obs"] = np.concatenate(data["obs"], axis=0)
        data["next_obs"] = np.concatenate(data["next_obs"], axis=0)
    data["act"] = np.concatenate(data["act"], axis=0)
    data["rwd"] = np.stack(data["rwd"])
    data["done"] = np.concatenate(data["done"], axis=0)
    return data


"""TODO: handle different gym versions more cleanly, maybe don't use agent.get_env()"""

class EMGWrapper(gym.Wrapper):
    def __init__(self, teacher, emg_simulator):
        super().__init__(teacher.get_env())
        self.teacher = teacher
        self.emg_simulator = emg_simulator

        # TODO fix this, values can be > 1 and < -1
        if emg_simulator.return_features:
            self.observation_space["observation"] = gym.spaces.Box(
                low=-10,
                high=10,
                shape=(emg_simulator.num_channels * 4,),
                dtype=np.float64
            )
        else:
            self.observation_space["observation"] = gym.spaces.Box(
                low=-3, 
                high=3, 
                shape=(emg_simulator.num_channels, emg_simulator.window_size),
                dtype=np.float64
            )

    def _obs_to_emg(self, state):
        ideal_action, _ = self.teacher.predict(state)
        # last action entry not used in fetch env
        out = self.emg_simulator(ideal_action)
        return out

    def reset(self):
        state = self.env.reset()
        state["emg_observation"] = self._obs_to_emg(state)
        return state

    def step(self, action):
        state, reward, terminated, info = self.env.step(action)
        state["emg_observation"] = self._obs_to_emg(state)
        return state, reward, terminated, info
    
    def get_ideal_action(self, state):
        short_state = {key: value for key, value in state.items() if key != "emg_observation"}
        return self.teacher.predict(short_state)


"""TODO: unify with EMGWrapper"""
class UserSimulator(gym.Wrapper):
    """Simulator maps intended action to emg and back to decoded action"""
    def __init__(self, env, decoder, config):
        super().__init__(env)
        self.decoder = decoder
        self.emg_simulator = WindowSimulator(num_actions=6,#FIXME make param
                                             num_bursts=config.n_bursts,
                                             num_channels=config.n_channels,
                                             window_size=config.window_size,
                                             noise=config.noise)

    def reset(self, seed=0):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        emg = self.emg_simulator(action[None])
        decoded_action = self.decoder.sample_action({'emg_observation': emg})[-1]
        state, reward, terminated, truncated, info = self.env.step(decoded_action)
        return state, reward, terminated, truncated, info
