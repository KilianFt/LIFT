import numpy as  np

def evaluate_emg_policy(env, model, n_eval_steps=1000, is_teacher=False):
    rewards = np.zeros(n_eval_steps)
    observation = env.reset()
    for i in range(n_eval_steps):
        if is_teacher:
            action, _ = model.predict(observation)
        else:
            action = model.sample_action(observation)
        observation, reward, terminated, info = env.step(action)
        rewards[i] = reward[-1]
    return rewards.mean()

def evaluate_teacher_policy(env, model, n_eval_steps=1000, is_teacher=False):
    actions = []
    rewards = np.zeros(n_eval_steps)
    observation, info = env.reset()
    for i in range(n_eval_steps):
        if is_teacher:
            action, _ = model.predict(observation)
        else:
            action = model.sample_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)        
        rewards[i] = reward
        actions.append(action)
    
    actions = np.stack(actions)
    return rewards.mean()
