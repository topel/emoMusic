__author__ = 'tpellegrini'

import csv
import numpy as np
import matplotlib.pyplot as plt

def load_sub(filename):
    res = dict()
    with (open(filename, 'rb')) as f:
        reader = csv.reader(f)
        headers = reader.next()
        # print headers
        for line in reader:
            res[line[0]] = np.array(line[1:])
    return res


indir='SUBMISSIONS'


song = '2027'
# song = '2045'
valence1 = load_sub(indir + '/1/' + 'me15em_IRIT-SAMOVA_rnn260feat_valence.csv')
print valence1.keys()
# print valence1[song][0:20]

valence2 = load_sub(indir + '/2/' + 'me15em_IRIT-SAMOVA_rnn2x260feat_valence.csv')
# print valence2[song][0:20]

valence3 = load_sub(indir + '/3/' + 'me15em_IRIT-SAMOVA_rnn268feat_valence.csv')
# print valence3[song][0:20]

valence4 = load_sub(indir + '/4/' + 'me15em_IRIT-SAMOVA_rnn2x268feat_valence.csv')
# print valence4[song][0:20]

valence5 = dict()
for s in valence2:
    v1 = np.array(valence2[s], dtype=float)
    # print v1
    v2 = np.array(valence4[s], dtype=float)
    valence5[s] = 0.5* np.add(v1,v2)


# wts = np.ones(47)*1./48
# wts = np.hstack((np.array([1./96]), wts, np.array([1./96])))

taille = 12
wts = np.ones(taille-1)*1./taille
wts = np.hstack((np.array([1./(2*taille)]), wts, np.array([1./(2*taille)])))
delay = (wts.shape[0]-1) / 2

valence6 = dict()
for s in valence5:
    valence6[s] = np.convolve(valence5[s], wts, mode='same')
    valence6[s][:delay] = valence5[s][:delay]
    valence6[s][-delay:] = valence5[s][-delay:]


fig, ax = plt.subplots()
# ax.plot(valence1[song], 'k', label='1')
ax.plot(valence2[song], 'r', label='2')
# ax.plot(valence3[song], 'b', label='3')
ax.plot(valence4[song], 'g', label='4')
ax.plot(valence5[song], 'y', label='5')
ax.plot(valence6[song], 'xk', label='6')
legend = ax.legend(loc='upper right', shadow=True)
plt.show()
