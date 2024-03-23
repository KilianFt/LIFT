import copy
import numpy as np
from tqdm import tqdm

def rollout(
        env, 
        agent, 
        n_steps=1000, 
        terminate_on_done=True, 
        reset_on_done=False, 
        random_pertube_prob=0.0, 
        action_noise=0.0,
    ):
    """
    Args:
        env: gym environment
        agent: agent class with sample_action method
        n_steps: number of rollout steps
        terminate_on_done: whether to break loop when env returns done
        reset_on_done: whether to reset when env returns done
        random_pertube_prob: probability of random pertubation of teacher actions. 
            If pertub, then random action in range [-1, 1] is taken.
        action_noise: noise added to teacher actions.

    Returns:
        data: dict with fields [obs, act, rwd, next_obs, done]. dict observations will be concatenated for each key.
    """
    data = {"obs": [], "act": [], "rwd": [], "next_obs": [], "done": []}

    obs = env.reset()
    
    bar = tqdm(range(n_steps), desc="Rollout", unit="item")
    while len(data["rwd"]) < n_steps:
        act = agent.sample_action(obs)

        # randomely pertube teacher actions to obtain broader diversity in data
        is_pertub = False
        if np.random.rand() < random_pertube_prob:
            act = np.random.rand(*act.shape) * 2 - 1
            is_pertub = True
        
        act += np.random.randn(*act.shape) * action_noise

        next_obs, rwd, done, info = env.step(act)

        if not is_pertub:
            data["rwd"].append(float(rwd))
            data["obs"].append(copy.deepcopy(obs))
            data["act"].append(act)
            data["next_obs"].append(copy.deepcopy(next_obs))
            data["done"].append(bool(done))
            bar.update(1)
        
        obs = next_obs

        if done and terminate_on_done:
            break
        if done and reset_on_done:
            obs = env.reset()
    
    env.close()

    if isinstance(data["obs"][0], dict):
        print("dict")
        keys = list(data["obs"][0].keys())
        print(keys)
        data["obs"] = {k: np.stack([o[k] for o in data["obs"]]) for k in keys}
        data["next_obs"] = {k: np.stack([o[k] for o in data["next_obs"]]) for k in keys}
    else:
        data["obs"] = np.stack(data["obs"])
        data["next_obs"] = np.stack(data["next_obs"])
    data["act"] = np.stack(data["act"])
    data["rwd"] = np.stack(data["rwd"])
    data["done"] = np.stack(data["done"])
    return data

if __name__ == "__main__":
    from lift.environments.gym_envs import NpGymEnv
    np.random.seed(0)

    class RandomPolicy:
        def __init__(self, env):
            self.env = env
        
        def sample_action(self, obs):
            return self.env.action_space.sample()
    
    # test rollout obs concat and terminate_on_done
    env = NpGymEnv("FetchReachDense-v2", cat_obs=False)
    random_policy = RandomPolicy(env)

    data = rollout(env, random_policy, n_steps=100)
    assert data["obs"]["observation"].shape[0] < 100
    assert list(data["obs"].keys()) == ["observation", "achieved_goal", "desired_goal"]
    assert sum(data["done"] == True) == 1

    # test reset_on_done
    data = rollout(env, random_policy, n_steps=100, terminate_on_done=False, reset_on_done=True)
    assert data["obs"]["observation"].shape == (100, 10)
    assert data["next_obs"]["observation"].shape == (100, 10)
    assert data["rwd"].shape == (100,)
    assert data["done"].shape == (100,)
    assert sum(data["done"] == True) > 1