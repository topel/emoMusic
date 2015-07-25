__author__ = 'thomas'

import essentia
from essentia.standard import *
from os import listdir
from os.path import basename, splitext

# we start by instantiating the audio loader:

FEATURE_DIR = '/baie/corpus/emoMusic/train/essentia_features/'
AUDIO_DIR = '/baie/corpus/emoMusic/train/audio/'

import matplotlib.pylab as plt

# plt.plot(audio[1*44100:2*44100])
# plt.show()

w = Windowing(type = 'blackmanharris62')

spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum

mfcc = MFCC()

# let's have a look at the inline help:
# help(MFCC)

# # extract MFCCs
# mfccs = []
# for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
#     mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
#     mfccs.append(mfcc_coeffs)
#
# # transpose to have it in a better shape
# # we need to convert the list to an essentia.array first (== numpy.array of floats)
# mfccs = essentia.array(mfccs).T
#
# # # and plot
# # plt.imshow(mfccs[1:,:], aspect = 'auto')
# # plt.show()

dc_algo = DynamicComplexity(frameSize = 0.5)

loudness_algo = Loudness()

danceability_algo = Danceability()

# silence_rate_algo = SilenceRate(thresholds= [0, 10, 30, 60])

spectral_complexity_algo = SpectralComplexity()

spectral_contrast_algo = SpectralContrast(frameSize = 22050)

# high-frequency content:
hfc_algo = HFC()

zcr_algo = ZeroCrossingRate()

barkbands_algo = BarkBands()

melbands_algo = MelBands()

erbbands_algo = ERBBands()

gfcc_algo = GFCC()

crest_algo = Crest()
flatnessdb_algo = FlatnessDB()
rms_algo = RMS()
flux_algo = Flux()
centroid_algo = Centroid()
central_moments_algo = CentralMoments()
roll_off_algo = RollOff()
decrease_algo = Decrease()

spectral_peaks_algo = SpectralPeaks()
dissonance_algo = Dissonance()

pitchsalience_algo = PitchSalience()

# extract MFCCs with a pool
pool = essentia.Pool()

for audio_filename in listdir(AUDIO_DIR):
    print audio_filename
    song_id = splitext(basename(audio_filename))[0]
    audio_filename =  AUDIO_DIR + '%s'%(audio_filename)
    output_filename = FEATURE_DIR + '%s.yaml'%(song_id)

    loader = EasyLoader(filename = audio_filename, startTime = 15)

    # and then we actually perform the loading:
    audio = loader()

    # for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
    for frame in FrameGenerator(audio, frameSize = 22050, hopSize = 22050):
        w_frame = w(frame)
        spectrum_frame = spectrum(w_frame)

        mfcc_bands, mfcc_coeffs = mfcc(spectrum_frame)
        dc, _ = dc_algo(w_frame)

        spectrum_rms = rms_algo(spectrum_frame)
        spectrum_flux = flux_algo(spectrum_frame)
        spectrum_centroid = centroid_algo(spectrum_frame)
        spectrum_rolloff = roll_off_algo(spectrum_frame)
        spectrum_decrease = decrease_algo(spectrum_frame)

        loudness = loudness_algo(w_frame)
        danceability = danceability_algo(w_frame)
        # silence_rate = silence_rate_algo(w_frame)
        spectral_contrast, spectral_valley = spectral_contrast_algo(spectrum_frame)
        hfc = hfc_algo(spectrum_frame)
        zcr = zcr_algo(w_frame)
        barkbands = barkbands_algo(spectrum_frame)
        melbands =  melbands_algo(spectrum_frame)
        erbbands = erbbands_algo(spectrum_frame)
        _, gfcc = gfcc_algo(spectrum_frame)

        # crest_spectrum = crest_algo(spectrum_frame)

        crest_bark = crest_algo(barkbands)
        flatnessdb_bark = flatnessdb_algo(barkbands)
        central_moments_bark = central_moments_algo(barkbands)

        crest_erb = crest_algo(erbbands)
        flatnessdb_erb = flatnessdb_algo(erbbands)
        central_moments_erb = central_moments_algo(erbbands)

        crest_mel = crest_algo(melbands)
        flatnessdb_mel = flatnessdb_algo(melbands)
        central_moments_mel = central_moments_algo(melbands)

        freq_peaks, freq_mag = spectral_peaks_algo(spectrum_frame)
        dissonance = dissonance_algo(freq_peaks, freq_mag)

        pitchsalience = pitchsalience_algo(spectrum_frame)

        spectral_complexity = spectral_complexity_algo(spectrum_frame)

        pool.add('lowlevel.dynamic_complexity', dc)
        pool.add('lowlevel.loudness', loudness)

        pool.add('lowlevel.spectrum_rms', spectrum_rms)
        pool.add('lowlevel.spectrum_flux', spectrum_flux)
        pool.add('lowlevel.spectrum_centroid', spectrum_centroid)
        pool.add('lowlevel.spectrum_rolloff', spectrum_rolloff)
        pool.add('lowlevel.spectrum_decrease', spectrum_decrease)

        pool.add('lowlevel.hfc', hfc)

        pool.add('lowlevel.zcr', zcr)

        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.mfcc_bands', mfcc_bands)

        pool.add('lowlevel.barkbands', barkbands)
        pool.add('lowlevel.crest_bark', crest_bark)
        pool.add('lowlevel.flatnessdb_bark', flatnessdb_bark)
        pool.add('lowlevel.central_moments_bark', central_moments_bark)

        pool.add('lowlevel.melbands', melbands)
        pool.add('lowlevel.crest_erb', crest_erb)
        pool.add('lowlevel.flatnessdb_erb', flatnessdb_erb)
        pool.add('lowlevel.central_moments_erb', central_moments_erb)

        pool.add('lowlevel.erbbands', erbbands)
        pool.add('lowlevel.crest_mel', crest_mel)
        pool.add('lowlevel.flatnessdb_mel', flatnessdb_mel)
        pool.add('lowlevel.central_moments_mel', central_moments_mel)

        pool.add('lowlevel.gfcc', gfcc)

        pool.add('lowlevel.spectral_contrast', spectral_contrast)
        pool.add('lowlevel.spectral_valley', spectral_valley)

        pool.add('lowlevel.dissonance', dissonance)

        pool.add('lowlevel.pitchsalience', pitchsalience)

        pool.add('lowlevel.spectral_complexity', spectral_complexity)

        pool.add('lowlevel.danceability', danceability)


    # print pool['lowlevel.mfcc'].shape
    # (61, 13)

    # plt.imshow(pool['lowlevel.mfcc'].T[1:,:], aspect = 'auto')
    # plt.show() # unnecessary if you started "ipython --pylab"
    # plt.figure()
    # plt.imshow(pool['lowlevel.mfcc_bands'].T, aspect = 'auto', interpolation = 'nearest')

    # or as a one-liner:
    YamlOutput(filename = output_filename)(pool)

