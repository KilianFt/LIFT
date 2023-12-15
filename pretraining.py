from collections import defaultdict

from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader, random_split
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from lift.datasets import EMGSLDataset
from lift.controllers import MLP, EMGPolicy, EMGAgent
from lift.evaluation import evaluate_emg_policy


def rollout(emg_env, config):
    observation = emg_env.reset()
    history = defaultdict(list)

    for _ in tqdm(range(config.n_steps_rollout), desc="Rollout", unit="item"):
        action, _ = emg_env.get_ideal_action(observation)
        observation, reward, terminated, info = emg_env.step(action)

        history['emg_observation'].append(observation["emg_observation"][-1])
        history['action'].append(action[-1])

        if terminated:
            observation = emg_env.reset()

    return history


def get_pretrain_dataloaders(history, config, train_percentage: float = 0.8):
    dataset = EMGSLDataset(obs=history['emg_observation'], action=history['action'])

    train_size = int(train_percentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)
    return train_dataloader, val_dataloader


def train_policy(emg_env, config):
    history = rollout(emg_env, config)

    logger = WandbLogger(log_model="all")

    train_dataloader, val_dataloader = get_pretrain_dataloaders(history, config)

    input_size = config.n_channels * config.window_size
    action_size = emg_env.action_space.shape[0]
    hidden_sizes = [config.hidden_size for _ in range(config.n_layers)]
    model = MLP(input_size=input_size, output_size=action_size, hidden_sizes=hidden_sizes)
    pl_model = EMGPolicy(lr=config.lr, model=model)
    agent = EMGAgent(policy=pl_model.model)

    trainer = L.Trainer(max_epochs=config.epochs, log_every_n_steps=1, check_val_every_n_epoch=1,
                        enable_checkpointing=False, logger=logger, gradient_clip_val=config.gradient_clip_val)
    trainer.fit(model=pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    mean_reward = evaluate_emg_policy(emg_env, agent)
    print(f"Pretrain reward {mean_reward}")
    wandb.log({'pretrain_reward': mean_reward})

    return agent
