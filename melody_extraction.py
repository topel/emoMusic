__author__ = 'thomas'

import numpy as np
import subprocess
from utils import load_y_to_dict
import time
from os import path

annotator = '/home/thomas/software/sonic-annotator-1.1/sonic-annotator'
plugin='vamp:mtg-melodia:melodia:melody'
feature_dir='/baie/corpus/emoMusic/train/vamp_features'
melody_dir = feature_dir + '/melody'
DATADIR = '/baie/corpus/emoMusic/train/'

data = dict()
arousal, valence, song_ids, nb_of_songs = load_y_to_dict(DATADIR)

# song_id=250
# for song_id in [250]:
doExtractMelody = True
if doExtractMelody:
    for song_id in song_ids:

        audio_file='/baie/corpus/emoMusic/train/audio/%d.mp3'%(song_id)
        output = melody_dir + '/%d_vamp_mtg-melodia_melodia_melody.csv'%(song_id)

        if path.isfile(output):
            print '%s already exists'%(output)
            continue
        # cline = [annotator, '-d', plugin, audio_file, '-w', 'csv', '--csv-stdout']
        # proc = Popen(cline, stdout=PIPE, stderr=PIPE)
        # feat, stdout_messages = proc.communicate()

        start_time = time.time()
        cline = [annotator, '-d', plugin, audio_file, '-w', 'csv', '--csv-basedir', melody_dir]
        subprocess.call(cline)
        # proc = Popen(cline, stdout=PIPE, stderr=PIPE)
        # _, stdout_messages = proc.communicate()

        print("--- Melody extraction (VAMP): %.2f seconds ---" % (time.time() - start_time))

