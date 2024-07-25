import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import numpy as np


def plot_timeseries(states):

    biweek_steps = states.shape[0]
    color_params = dict(c=range(biweek_steps), cmap="viridis")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.scatter(states[:, 0], states[:, 1], **color_params)
    ax.set(xlabel="S(t)", ylabel="I(t)", xscale="log", yscale="log")
    fig.set_tight_layout(True)

    fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
    tt = [x/26. for x in range(biweek_steps)]
    Nt = states.sum(axis=1)
    axs[0].scatter(tt, states[:, 1], **color_params)
    axs[0].set(ylabel="I(t)")
    axs[1].scatter(tt, states[:, 0]/Nt, **color_params)
    axs[1].set(ylabel="S(t)/N")
    axs[2].scatter(tt, Nt, **color_params)
    axs[2].set(xlabel="t [years]", ylabel="N")
    fig.set_tight_layout(True)


def plot_animation(states, settlements_df, save_path=None):

    fig, ax = plt.subplots()

    scat = ax.scatter(
        settlements_df.Long, 
        settlements_df.Lat, 
        s=0.1*np.sqrt(settlements_df.population), 
        c=states[0, :, 1] / states[0, :, :].sum(axis=-1), 
        cmap="Reds", norm=LogNorm(vmin=1e-4, vmax=0.01), alpha=0.5)

    def animate(i):
         ax.set_title("{:.2f} years".format(i/26.))
         scat.set_array(states[i, :, 1] / states[i, :, :].sum(axis=-1))
         return scat,

    ani = animation.FuncAnimation(fig, animate, frames=states.shape[0]-1, interval=50, blit=False)

    # To save the animation using Pillow as a gif
    if save_path is not None:

        writer = animation.PillowWriter(fps=15,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
        
        ani.save(save_path, writer=writer)

    return ani