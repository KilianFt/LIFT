import os
import torch
from configs import BaseConfig
from lift.environments.simulator import WindowSimulator

def main():
    torch.manual_seed(0)

    config = BaseConfig()
    sim = WindowSimulator(
        action_size=config.action_size,
        num_bursts=config.simulator.n_bursts,
        num_channels=config.n_channels,
        window_size=config.window_size,
        return_features=False
    )
    sim.fit_params_to_mad_sample(
        (config.mad_data_path / "Female0"/ "training0").as_posix()
    )

    # simulate 1 dof at a time
    single_actions = torch.tensor([
        [0, 0, 0],
        [0.5, 0, 0], 
        [-0.5, 0, 0], 
        [0, 0.5, 0], 
        [0, -0.5, 0], 
        [0, 0, 0.5], 
        [0, 0, -0.5]
    ])
    
    num_samples = 30
    data = {"emg": [], "action": []}
    for single_action in single_actions:
        actions = single_action.view(1, -1).repeat_interleave(num_samples, 0)
        data["emg"].append(sim(actions))
        data["action"].append(single_action)
    
    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path)
    torch.save(data, os.path.join(config.data_path, "fake_mad_data.pt"))

if __name__ == "__main__":
    main()