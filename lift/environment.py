import gymnasium as gym
import numpy as np

from lift.simulator import EMGSimulator


class EMGWrapper(gym.Wrapper):
    def __init__(self, teacher, config):
        super().__init__(teacher.get_env())
        self.teacher = teacher
        self.emg_simulator = EMGSimulator(n_channels=config.n_channels, window_size=config.window_size)
        self.observation_space["observation"] = gym.spaces.Box(low=-1, high=1,
                                                               shape=(config.n_channels, config.window_size),
                                                               dtype=np.float64)

    def _obs_to_emg(self, state):
        ideal_action, _ = self.teacher.predict(state)
        return self.emg_simulator(ideal_action)

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


class ActionNoiseWrapper(gym.Wrapper):
    def __init__(self, env, noise_level=0., add_task_id=False):
        super().__init__(env)
        self.noise_level = noise_level
        self.add_task_id = add_task_id

        low = self.observation_space["observation"].low
        high = self.observation_space["observation"].high
        shape = self.observation_space["observation"].shape

        if add_task_id:
            low = np.insert(low, -1, low[-1])
            high = np.insert(high, -1, high[-1])
            shape = (shape[0] + 1,)

        self.observation_space["observation"] = gym.spaces.Box(
            low=low, high=high, shape=shape,
        ) # reserve last observation
    
    def _append_task_id(self, state):
        if self.add_task_id:
            task_id = self.noise_level * np.ones_like(state[..., :1])
            return np.concatenate([state, task_id], axis=-1)
        else:
            return state
    
    def _add_action_noise(self, action):
        eps = np.random.normal(action.shape)
        action_ = action + self.noise_level * eps
        return action_
    
    def reset(self, *args, **kwargs):
        state, info = self.env.reset()
        state["observation"] = self._append_task_id(state["observation"])
        return state, info
    
    def step(self, action):
        action_ = self._add_action_noise(action)
        state, reward, terminated, truncated, info = self.env.step(action_)
        state["observation"] = self._append_task_id(state["observation"])
        return state, reward, terminated, truncated, info
    
if __name__ == "__main__":
    np.random.seed(0)
    env = ActionNoiseWrapper(
        gym.make("FetchReachDense-v2"),
        noise_level=1.,
        add_task_id=True,
    )

    obs, _ = env.reset()
    next_obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs["observation"].shape == env.observation_space["observation"].shape
    assert next_obs["observation"].shape == env.observation_space["observation"].shape