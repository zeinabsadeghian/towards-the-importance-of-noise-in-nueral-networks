import numpy as np
import torch


def random_ep(p_a, p_w, a_size, weights_size):
    big_ep = np.random.uniform(0, p_w, weights_size)
    small_ep = np.random.uniform(0, p_a, a_size)
    big_ep = big_ep.astype("float32")
    small_ep = small_ep.astype("float32")
    final_big_epsilon_tensor = torch.from_numpy(big_ep)
    final_big_epsilon_tensor = final_big_epsilon_tensor / torch.norm(final_big_epsilon_tensor)
    final_small_epsilon_tensor = torch.from_numpy(small_ep)
    final_small_epsilon_tensor = final_small_epsilon_tensor / torch.norm(final_small_epsilon_tensor)

    return [final_big_epsilon_tensor, final_small_epsilon_tensor]