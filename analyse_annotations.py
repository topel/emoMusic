__author__ = 'thomas'

import numpy as np
from utils import load_X, load_y_to_dict, mix, standardize, add_intercept, evaluate, evaluate1d, load_metadata
import matplotlib.pyplot as plt

NUM_FRAMES = 60
DATADIR = '/baie/corpus/emoMusic/train/'
# DATADIR = './train/'

arousal, valence, song_id, nb_of_songs = load_y_to_dict(DATADIR)

metadatafile = DATADIR + 'annotations/metadata.csv'
list_20genres_file = DATADIR + '../20_most_frequent_genres.lst'
id2genre, genre_list = load_metadata(metadatafile, list_20genres_file)

# for k, v in id2genre.iteritems():
#     print k, v
#
# for g in genre_list:
#     print g
#
# for song in song_id:
#     print song, arousal[song]

arousal_by_genre = dict()
valence_by_genre = dict()


for genre in genre_list:
    arousal_tmp = list()
    valence_tmp = list()
    for id, g in id2genre.iteritems():
        if g == genre:
            arousal_tmp.append(arousal[id])
            valence_tmp.append(valence[id])
    # arousal_by_genre[genre] = arousal_tmp
    # valence_by_genre[genre] = valence_tmp
    arousal_by_genre[genre] = [item for sublist in arousal_tmp for item in sublist]
    valence_by_genre[genre] = [item for sublist in valence_tmp for item in sublist]

ind = 1
for genre in genre_list:
    print genre
    val = valence_by_genre[genre]
    ar = arousal_by_genre[genre]
    plt.figure(ind)
    plt.plot(val, ar, 'o')
    plt.title(genre)
    axes = plt.gca()
    axes.set_xlim([-1.,1.])
    axes.set_ylim([-1.,1.])
    plt.show()
    # if ind % 5 == 0 :
    #     break
    ind += 1