import torch.nn as nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage

def make_collector(config, train_env, actor_model_explore):
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=config.init_random_frames,
        frames_per_batch=config.frames_per_batch,
        total_frames=config.total_frames,
        device=config.device,
    )
    collector.set_seed(config.seed)
    return collector

"""TODO: maybe unify offline and online replay buffer"""
def make_replay_buffer(
    batch_size,
    prioritize=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    if prioritize:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer

def get_activation(activation):
    if activation == "relu":
        return nn.ReLU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "silu":
        return nn.SiLU
    else:
        raise NotImplementedError
    
def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)