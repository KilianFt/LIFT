import numpy as  np
from gymnasium.utils.step_api_compatibility import step_api_compatibility

def evaluate_emg_policy(env, model, n_eval_steps=1000, is_teacher=False):
    rewards = np.zeros(n_eval_steps)
    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]
    
    for i in range(n_eval_steps):
        if is_teacher:
            action, _ = model.predict(observation)
        else:
            action = model.sample_action(observation)
        # observation, reward, terminated, info = env.step(action)
        # rewards[i] = reward[-1]
        
        observation, reward, terminated, info = step_api_compatibility(
            env.step(action), 
            output_truncation_bool=False
        )
        rewards[i] = float(reward)
    return rewards.mean()
