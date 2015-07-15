__author__ = 'thomas'

# Load modules and data
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import cPickle as pickle

# spector_data = sm.datasets.spector.load()
# spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)
#
# # Fit and summarize OLS model
# mod = sm.OLS(spector_data.endog, spector_data.exog)
# res = mod.fit()
# print res.summary()

from utils import load_X, load_y, mix, standardize, add_intercept, evaluate, evaluate1d
import matplotlib.pyplot as plt

def load_X_from_fold(fold, subset):
    song_ids = list()
    X_ = None
    y_ = None
    for k, v in fold[subset].iteritems():
        if (k == 'std' or k == 'mean'):
            continue
        song_ids.append(k)
        val = np.array(v['valence'], dtype=float)
        ar = np.array(v['arousal'], dtype=float)
        tmp = np.hstack((val[:,np.newaxis], ar[:,np.newaxis]))

        if X_ is None:
            X_ = v['X']
            y_ = tmp
        else:
            X_ = np.vstack((X_, v['X']))
            y_ = np.vstack((y_, tmp))
    return X_,  y_, np.array(song_ids, dtype=int)



PURCENT = 5 # Purcentage of the set you want on the test set
NUM_FRAMES = 60
DATADIR = '/baie/corpus/emoMusic/train/'
# DATADIR = './train/'


EMO='valence'
# EMO='arousal'
do_regularize = False

fold_id = 2
fold0 = pickle.load( open( DATADIR + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )

X_train, y_train, id_train = load_X_from_fold(fold0, 'train')
X_test, y_test, id_test = load_X_from_fold(fold0, 'test')

print id_test.shape

X_train = X_train[:,[10,12,13,17,19,82,83,84,85,89,90,91,103,140,142,146,148,212,214,218,220]]
X_test = X_test[:,[10,12,13,17,19,82,83,84,85,89,90,91,103,140,142,146,148,212,214,218,220]]
# X_train = X_train[:,[13,85,103,142,214]]
# X_test = X_test[:,[13,85,103,142,214]]

# one dimension at a time
# 0: arousal, 1: valence
if EMO == 'valence':
    print '... emotion: valence'
    y_train = y_train[:,0]
    y_test = y_test[:,0]
else:
    print '... emotion: arousal'
    y_train = y_train[:,1]
    y_test = y_test[:,1]

# X_test = X_train[119:119+y_test.shape[0],:]
# y_test = y_train[119:119+y_test.shape[0]]

print X_train.shape, y_train.shape, X_test.shape, y_test.shape

# add column of ones to data to account for the bias:
X_train = add_intercept(X_train)
print X_train.shape
# print X_train[0:10]

# Fit regression model
res = sm.OLS(y_train, X_train).fit()
# Inspect the results
print res.summary()

# cf http://statsmodels.sourceforge.net/devel/mixed_linear.html
# md = smf.mixedlm(y_train, X_train, groups=data["Pig"])
# mdf = md.fit()
# print(mdf.summary())


X_test = add_intercept(X_test)
pred = res.predict(X_test)
print pred

y_hat = np.array(pred, dtype=float)

RMSE, pcorr, error_per_song, mean_per_song = evaluate1d(y_test, y_hat, id_test.shape[0])

All_stack =np.hstack(( error_per_song, mean_per_song ))
print'  Error per song (ar/val)  Mean_per_song (ar/val)    :\n'
print(All_stack)
print '\n'

print'song_id :'
print(id_test)
print '\n'
#print('Error per song: \n', Error_per_song)

print(
        'sklearn --> arrousal : %.4f, valence : %.4f\n'
        'Pearson Corr --> arrousal : %.4f, valence : %.4f \n'
        % (RMSE[0], -1. , pcorr[0][0], -1)
      # % (RMSE[0],RMSE[1],pcorr[0][0], pcorr[1][0])
)

fig, ax = plt.subplots()
x1 = np.linspace(1,len(pred),len(pred))
ax.plot(x1, y_test, 'o', label="Data")
ax.plot(x1, y_hat, 'r-', label="OLS prediction")
plt.title(EMO + ' on Test subset')
ax.legend(loc="best")
# plt.show()
plt.savefig('figures/linreg_sm_%s_fold%d.png'%(EMO, fold_id), format='png')
