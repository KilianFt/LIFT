import copy
import numpy as np
from gymnasium.utils.step_api_compatibility import step_api_compatibility

from lift.utils import obs_wrapper

"""TODO: still not handeling raw gym env correctly since we are assuming we always use teacher.get_env(). 
raw env does not add batch dimension to observations which makes data aggregation at the end incorrect"""
def evaluate_policy(env, policy, eval_steps=1000, use_terminate=True, is_sb3=False):
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
    rewards = np.zeros((eval_steps,))

    observation = obs_wrapper(env.reset())
    
    for t in range(eval_steps):
        if is_sb3:
            action, _ = policy.predict(observation)
        else:
            action = policy.sample_action(observation)
        
        next_observation, reward, terminated, info = step_api_compatibility(
            env.step(action), 
            output_truncation_bool=False
        )
        observation = next_observation

        rewards[t] = float(reward)

        if use_terminate and terminated:
            break
        
        if terminated:
            observation = obs_wrapper(env.reset())
    
    return rewards
