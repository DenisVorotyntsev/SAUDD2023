import os

import time
import numpy as np
import matplotlib.pyplot as plt


def plot_gradients(model, output_folder):
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.cpu().detach().numpy()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Plot and save gradient distribution for each layer
    for name, grad in gradients.items():
        plt.hist(np.reshape(grad, [-1]))
        plt.title(f"{name} Gradient Distribution")
        plt.xlabel("Gradient Bins")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_folder, f"{name}.png"))
        plt.clf()


class Clock:
    def __init__(self, timeout):
        self.timeout = timeout

    def tik(self):
        self.t_start = time.time()

    def tok(self):
        time_left = self.timeout - (time.time() - self.t_start)
        return time_left
