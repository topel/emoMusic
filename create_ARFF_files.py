__author__ = 'thomas'

import cPickle as pickle
# input = '/baie/corpus/emoMusic/test/pkl/test_set_baseline_268features_58songs_normed.pkl'
# output = '/baie/corpus/emoMusic/test/test_set_baseline_268features_58songs_normed.arff'
input = '/baie/corpus/emoMusic/train/pkl/train_set_baseline_268features_431songs_normed.pkl'
output = '/baie/corpus/emoMusic/train/train_set_baseline_268features_431songs_normed.arff'

isTest = False

data = pickle.load( open( input, "rb" ) )

header = (
    '% IRIT-SAMOVA additional features used for the Emotion in Music task 2015\n'
    '@RELATION emoMusic\n'
    '\n'
    '@ATTRIBUTE song_id NUMERIC\n'
    '@ATTRIBUTE timestamp NUMERIC\n'
    '@ATTRIBUTE flatnessdB_bark  NUMERIC\n'
    '@ATTRIBUTE flatnessdB_erb  NUMERIC\n'
    '@ATTRIBUTE spectral_valley_b1  NUMERIC\n'
    '@ATTRIBUTE spectral_valley_b2  NUMERIC\n'
    '@ATTRIBUTE spectral_valley_b3  NUMERIC\n'
    '@ATTRIBUTE spectral_valley_b4  NUMERIC\n'
    '@ATTRIBUTE spectral_valley_b5  NUMERIC\n'
    '@ATTRIBUTE spectral_valley_b6  NUMERIC\n'
    '\n'
    '@DATA\n'
)

with open(output, 'w') as f:
    f.write(header)
    if isTest:
    # print header
        for id, v in data.iteritems():
            for i in range(v['X'].shape[0]):
                temps = 15. + i * 0.5
                s = '%s,%.1f,'%(id, temps)
                s += ','.join(map(str, v['X'][i,-8:]))
                s += '\n'
                f.write(s)
    else:
        ind_song = 0
        for id in data['train']['song_id']:
            for i in range(data['train']['X'][ind_song].shape[0]):
                temps = 15. + i * 0.5
                s = '%s,%.1f,'%(id, temps)
                s += ','.join(map(str, data['train']['X'][ind_song][i,-8:]))
                s += '\n'
                f.write(s)
            ind_song += 1
