__author__ = 'tpellegrini'

from utils import load_y_to_dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

nb_points_to_plot = 3
x = np.arange(0, nb_points_to_plot, 1)        # x-array
gtdir = 'train/'
arousal, valence, song_id, nb_of_songs = load_y_to_dict(gtdir)
# print arousal.keys()

song = '1284'
duration = len(arousal[song])
print duration

ar = np.array(arousal[song])
val = np.array(valence[song])

line, = ax.plot(x, ar[x])
line_val, = ax.plot(x, val[x])
plt.ylim([-1., 1.])

def animate(i):
    print i, x+i
    line.set_ydata(ar[x+i])  # update the data
    line_val.set_ydata(val[x+i])  # update the data
    return line, line_val,

#Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    line_val.set_ydata(np.ma.array(x, mask=True))
    return line, line_val,

ani = animation.FuncAnimation(fig, animate, np.arange(1, duration-nb_points_to_plot+1), init_func=init,
    interval=100, blit=False)
plt.show()
