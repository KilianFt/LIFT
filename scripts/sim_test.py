import torch
from torch.utils.data import DataLoader, random_split
import lightning as L 
from pytorch_lightning.loggers import WandbLogger

from lift.datasets import EMGSLDataset
from lift.controllers import EMGPolicy, GaussianEncoder
from lift.simulator.simulator import WindowSimulator
from configs import BaseConfig


def get_dataloaders(observations, actions, train_percentage=0.8, batch_size=32):
    dataset = EMGSLDataset(obs=observations, action=actions)

    train_size = int(train_percentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader


def train(sim, actions, logger, config):
    features = sim(actions)
    train_dataloader, val_dataloader = get_dataloaders(features, actions, batch_size=config.batch_size)

    hidden_sizes = [config.encoder.hidden_size for _ in range(config.encoder.n_layers)]
 
    model = GaussianEncoder(input_dim=config.feature_size,
                    output_dim=config.action_size,
                    hidden_dims=hidden_sizes,)

    pl_model = EMGPolicy(lr=config.lr, model=model)
    trainer = L.Trainer(
        max_epochs=config.epochs, 
        log_every_n_steps=1, 
        check_val_every_n_epoch=1,
        enable_checkpointing=False, 
        gradient_clip_val=config.gradient_clip_val,
        logger=logger,
    )
    trainer.fit(model=pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    val = trainer.validate(model=pl_model, dataloaders=val_dataloader)
    return val


def main():
    L.seed_everything(100)

    config = BaseConfig()

    data_path = './datasets/MyoArmbandDataset/PreTrainingDataset/Female0/training0/'
    sim = WindowSimulator(action_size=config.action_size, num_bursts=config.simulator.n_bursts, num_channels=config.n_channels,
                          window_size=config.window_size, return_features=True)
    sim.fit_params_to_mad_sample(data_path)
    sim.fit_normalization_params()

    logger = WandbLogger(project='lift', tags='sim_testing')

    # one action at a time
    single_actions = torch.tensor([[0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0], [0, 0, 0.5], [0, 0, -0.5]])
    actions = single_actions.repeat_interleave(repeats=10, dim=0)
    single_val = train(sim, actions, logger, config)

    # test on combined simlutaneous actions
    base_actions = torch.tensor([0.5, -0.5])
    raw_actions = torch.cartesian_prod(base_actions, base_actions, base_actions)

    actions = raw_actions.repeat_interleave(repeats=10, dim=0)
    simul_val = train(sim, actions, logger, config)

    # test on random actions
    actions = torch.rand(128, 3) * 2 - 1
    random_val = train(sim, actions, logger, config)

    print(f"val loss\nsingle: {single_val[-1]['val_loss']:.4f}\nsimultaneous: {simul_val[-1]['val_loss']:.4f}\nrandom: {random_val[-1]['val_loss']:.4f}")


if __name__ == '__main__':
    main()
