import copy
import numpy as np
from gymnasium.utils.step_api_compatibility import step_api_compatibility

def evaluate_emg_policy(env, model, n_eval_steps=1000, use_terminate=True, is_teacher=False, save_data=True):
    data = {"obs": [], "act": [], "rwd": [], "next_obs": [], "done": []}

    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]
    
    for i in range(n_eval_steps):
        if is_teacher:
            action, _ = model.predict(observation)
        else:
            action = model.sample_action(observation)
        
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
