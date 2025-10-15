import os

import numpy as np
import pandas as pd
import torch
import random

from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

from utils.tools import setSeed


def create_synthetic_dataset(N, N_input, N_output, sigma):
    # N: number of samples in each split (train, test)
    # N_input: import of time steps in input series
    # N_output: import of time steps in output series
    # sigma: standard deviation of additional noise
    X = []
    breakpoints = []
    for k in range(2 * N):
        serie = np.array([sigma * random.random() for i in range(N_input + N_output)])

        i1 = random.randint(1, 10)
        i2 = random.randint(10, 18)
        j1 = random.random()
        j2 = random.random()
        interval = abs(i2 - i1) + random.randint(-3, 3)
        serie[i1:i1 + 1] += j1
        serie[i2:i2 + 1] += j2
        serie[i2 + interval:] += (j2 - j1)
        X.append(serie)
        breakpoints.append(i2 + interval)
    X = np.stack(X)
    breakpoints = np.array(breakpoints)
    # return X[0:N,0:N_input], X[0:N, N_input:N_input+N_output], X[N:2*N,0:N_input], X[N:2*N, N_input:N_input+N_output],breakpoints[0:N], breakpoints[N:2*N]
    os.makedirs("../dataset/syn_data", exist_ok=True)
    pd.DataFrame(X).to_csv(f'../dataset/syn_data/syn_data_{N_input}_{N_output}.csv', header=False, index=False)
    print()


if __name__ == '__main__':
    setSeed(2024)
    create_synthetic_dataset(500, 20, 20, 1)
