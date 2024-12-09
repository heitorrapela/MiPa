import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def lambda_(p, gamma = 0.05):
    return (2/(math.exp(-gamma*p)+1)) - 1

if __name__ == "__main__":
    epoch = 12
    gamma = 0.1
    sns.set(style='whitegrid')
    x = np.array(list(range(epoch)))
    y = np.array([lambda_(p, gamma=gamma) for p in x])
    sns.scatterplot(x=x, y=y)
    plt.savefig(f'f{epoch}_gamma_{gamma}.png')
