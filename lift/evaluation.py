import copy
import numpy as np
from gymnasium.utils.step_api_compatibility import step_api_compatibility

"""TODO: still not handeling raw gym env correctly since we are assuming we always use teacher.get_env(). 
raw env does not add batch dimension to observations which makes data aggregation at the end incorrect"""
def evaluate_policy(env, policy, eval_steps=1000, use_terminate=True, is_sb3=False, save_data=True):
    """
    Args:
        env: gym environment
        policy: policy class with sample_action method or predict method if stable_baseline3 policy.
        eval_steps: number of evaluation steps. Be careful whether the environment automatically resets. 
            fetch resets when done.
        use_terminate: whether to terminate when done.
        is_sb3: whether the policy is stable_baseline3 policy.
        save_data: whether to save trajectory data. If no, only save reward and other fields will be empty.

    Returns:
        data: dict with fields [obs, act, rwd, next_obs, done]. dict observations will be concatenated for each key.
    """
    data = {"obs": [], "act": [], "rwd": [], "next_obs": [], "done": []}

    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]
    
    for t in range(eval_steps):
        if is_sb3:
            action, _ = policy.predict(observation)
        else:
            action = policy.sample_action(observation)
        
        next_observation, reward, terminated, info = step_api_compatibility(
            env.step(action), 
            output_truncation_bool=False
        )

        data["rwd"].append(float(reward))
        if save_data:
            data["obs"].append(copy.deepcopy(observation))
            data["act"].append(action)
            data["next_obs"].append(copy.deepcopy(next_observation))
            data["done"].append(terminated)
        
        if use_terminate and terminated:
            break
        observation = next_observation
    
    if save_data: 
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
