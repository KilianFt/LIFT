import random
import torch


# class WeightedInterpolator:
#     def __init__(self, features, actions, k=None, sample=False):
#         self.features = features
#         self.actions = actions
#         self.epsilon = 1e-5
#         self.k = k
#         self.sample = sample

#     def __call__(self, new_actions):
#         if not isinstance(new_actions, torch.Tensor):
#             new_actions = torch.tensor(new_actions, dtype=torch.float32)
#         # Step 1: Calculate distances for each new action
#         # This results in a (batch_size, num_samples) distance matrix
#         distances = torch.norm(self.actions - new_actions[:, None, :], dim=2)

#         # Step 2: Compute interpolation weights
#         weights = 1 / (distances + self.epsilon)
#         weights /= weights.sum(axis=1, keepdims=True)

#         # Step 4: select k elements of the weights, set rest to 0
#         if self.k is not None:
#             if self.sample:
#                 indices = torch.multinomial(weights, self.k, replacement=False)
#             else:
#                 indices = torch.topk(weights, k=self.k, dim=1).indices

#             mask = torch.zeros_like(weights)
#             mask.scatter_(1, indices, 1)

#             sampled_weights = weights * mask
#             sampled_weights /= sampled_weights.sum(dim=1, keepdim=True)

#             weights = sampled_weights

#         # Step 3: Interpolate features
#         interpolated_features_batch = torch.tensordot(weights, self.features, dims=([1],[0]))
#         return interpolated_features_batch


class WeightedInterpolator:
    def __init__(self, features, actions, k=None, sample=False):
        self.features = features
        self.actions = actions
        self.epsilon = 1e-5
        self.k = k
        self.sample = sample

        self.unique_actions = torch.unique(actions, dim=0)

        # get mapping from unique indices to action indices
        self.u_act_to_actions = {}
        for i, u_act in enumerate(self.unique_actions):
            u_act_idxs = torch.argwhere((actions==u_act).all(dim=1))
            self.u_act_to_actions[i] = u_act_idxs.squeeze()

        assert actions.shape[0] == sum([x.shape[0] for x in self.u_act_to_actions.values()]), 'Number of action missmatch'

    def __call__(self, new_actions):
        if not isinstance(new_actions, torch.Tensor):
            new_actions = torch.tensor(new_actions, dtype=torch.float32)

        # Step 1: Calculate distances for each new action for all actions and unique actions
        unique_distances = torch.norm(self.unique_actions - new_actions[:, None, :], dim=2)
        distances = torch.norm(self.actions - new_actions[:, None, :], dim=2)

        # Step 2: Compute interpolation weights
        weights = 1 / (distances + self.epsilon)
        weights /= weights.sum(axis=1, keepdims=True)

        unique_weights = 1 / (unique_distances + self.epsilon)
        unique_weights /= unique_weights.sum(axis=1, keepdims=True)

        # Step 3: Ensure uniqueness of sampled actions
        if self.k is not None:
            if self.sample:
                indices = torch.multinomial(unique_weights, self.k, replacement=False)
            else:
                indices = torch.topk(unique_weights, k=self.k, dim=1).indices

            sampled_indices = torch.tensor([[random.choice(self.u_act_to_actions[val.item()]) for val in row] for row in indices])

            mask = torch.zeros(new_actions.shape[0], self.actions.shape[0])
            mask.scatter_(1, sampled_indices, 1)

            sampled_weights = weights * mask
            sampled_weights /= sampled_weights.sum(dim=1, keepdim=True)

            weights = sampled_weights

        # Step 4: Interpolate features
        interpolated_features_batch = torch.tensordot(weights, self.features, dims=([1],[0]))

        return interpolated_features_batch
