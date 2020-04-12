import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

x = np.array([4.59721805, 3.8335069 , 5.47918766, 6.78601795, 5.36538638,
       5.91973576, 4.08966419, 5.02783593, 2.9475715 , 5.38082603,
       4.23828663, 4.92882441, 6.21590435, 4.13167744, 6.73003558,
       4.40489628, 5.82815372, 3.90768045, 4.68658893, 5.05875578,
       4.43944884, 4.40811728, 5.82023621, 4.65540384, 5.29362033,
       5.46753602, 4.38395546, 5.3119342 , 5.98686689, 6.33043066,
       4.78698334, 4.91463257, 2.63910658, 6.85049807, 5.15989203,
       2.84519075, 5.14320164, 3.82806906, 5.21416232, 5.32820414,
       6.41239135, 3.14628898, 5.55714123, 3.64756411, 4.92984034,
       5.42729474, 4.98272126, 3.97479233, 5.84100994, 6.93632936])


def plot_data():
    fig, ax = plt.subplots(figsize=(12, 3))

    ax.scatter(x, [0] * len(x), alpha=.5, color='g')

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks(range(1, 10))
    ax.set_xlabel('Largura [cm]')

    plt.show()
        
    print('\nColected data:')

    return x


def plot_gaussian(mu=0, sigma=1):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = stats.norm.pdf(x, mu, sigma)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, y, color='g')
    ax.axvline(0, ls='--', lw=1, color='k')

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    plt.show()

    return


def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.e ** (0.5 * ((x - mu) / sigma) ** 2)


def likelihood(x, theta):
    n = len(x)
    mu, sigma = theta
    l = np.sum(np.log(gaussian(x, mu, sigma))) / n
    return l


def plot_gaussians_over_data():
    fig, (ax_u, ax_d) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax_d2 = ax_d.twinx()

    ax_d.scatter(x, [0] * len(x), alpha=.5, color='g')

    params = (
        [3, 0.5, 'r'], 
        [4, 0.5, 'b'], 
        [5, 0.5, 'g'], 
        [6, 0.5, 'y'], 
        [7, 0.5, 'k']
    )

    for mu, sigma, color in params:
        l = likelihood(x, (mu, sigma))
        ax_u.scatter(mu, l, c=color)

        x_ = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y_ = stats.norm.pdf(x_, mu, sigma)
        ax_d2.axvline(mu, ls='--', lw=1, c=color)
        ax_d2.plot(x_, y_, alpha=.3, color=color)

    for ax in [ax_u, ax_d, ax_d2]:
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks(range(1, 10))

    ax_d.set_xlabel('Largura [cm]')

    plt.show()
    
    return

