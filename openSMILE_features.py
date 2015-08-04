__author__ = 'thomas'

feature_names= ['F0final', 'voicingFinalUnclipped', 'jitterLocal', 'jitterDDP', 'shimmerLocal', 'logHNR', 'audspec_lengthL1norm', 'audspecRasta_lengthL1norm',
                'pcm_RMSenergy', 'pcm_zcr', 'audSpec_Rfilt', 'pcm_fftMag_fband250-650', 'pcm_fftMag_fband1000-4000', 'pcm_fftMag_spectralRollOff25.0', 'pcm_fftMag_spectralRollOff50.0', 'pcm_fftMag_spectralRollOff75.0', 'pcm_fftMag_spectralRollOff90.0',
                'pcm_fftMag_spectralFlux', 'pcm_fftMag_spectralCentroid', 'pcm_fftMag_spectralEntropy', 'pcm_fftMag_spectralVariance', 'pcm_fftMag_spectralSkewness', 'pcm_fftMag_spectralKurtosis', 'pcm_fftMag_spectralSlope', 'pcm_fftMag_psySharpness', 'pcm_fftMag_spectralHarmonicity',
                'pcm_fftMag_mfcc',
                'F0final_sma_de', 'voicingFinalUnclipped_sma_de', 'jitterLocal_sma_de', 'jitterDDP_sma_de', 'shimmerLocal_sma_de', 'logHNR_sma_de', 'audspec_lengthL1norm_sma_de', 'audspecRasta_lengthL1norm_sma_de', 'pcm_RMSenergy_sma_de', 'pcm_zcr_sma_de',
                'audSpec_Rfilt_sma_de',
                'pcm_fftMag_fband250-650_sma_de', 'pcm_fftMag_fband1000-4000_sma_de', 'pcm_fftMag_spectralRollOff25.0_sma_de', 'pcm_fftMag_spectralRollOff50.0_sma_de', 'pcm_fftMag_spectralRollOff75.0_sma_de', 'pcm_fftMag_spectralRollOff90.0_sma_de',
                'pcm_fftMag_spectralFlux_sma_de', 'pcm_fftMag_spectralCentroid_sma_de', 'pcm_fftMag_spectralEntropy_sma_de', 'pcm_fftMag_spectralVariance_sma_de', 'pcm_fftMag_spectralSkewness_sma_de', 'pcm_fftMag_spectralKurtosis_sma_de', 'pcm_fftMag_spectralSlope_sma_de', 'pcm_fftMag_psySharpness_sma_de', 'pcm_fftMag_spectralHarmonicity_sma_de',
                'pcm_fftMag_mfcc_sma_de']

openSMILE_feat_indices = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12], [12, 14], [14, 16],
    [16, 18], [18, 20], [20, 72], [72, 74], [74, 76], [76, 78], [78, 80], [80, 82], [82, 84],
    [84, 86], [86, 88], [88, 90], [90, 92], [92, 94], [94, 96], [96, 98], [98, 100], [100, 102],
    [102, 130],
    [130, 132], [132, 134], [134, 136], [136, 138], [138, 140], [140, 142], [142, 144], [144, 146], [146, 148], [148, 150],
    [150, 202],
    [202, 204], [204, 206], [206, 208], [208, 210], [210, 212], [212, 214],
    [214, 216], [216, 218], [218, 220], [220, 222], [222, 224], [224, 226], [226, 228], [228, 230], [230, 232],
    [232, 260]]

# awk '($3<0.2608 && $6<0.2457){print}' retrait_baseline_features.log
# allfolds valence: 0.2607 0.6753 arousal: 0.2453 0.6716 deb:14, fin:16 p_values: 0.994 0.953
# allfolds valence: 0.2606 0.6753 arousal: 0.2453 0.6717 deb:74, fin:76 p_values: 0.825 0.902
# allfolds valence: 0.2607 0.6749 arousal: 0.2454 0.6713 deb:80, fin:82 p_values: 0.793 0.847
# allfolds valence: 0.2607 0.6750 arousal: 0.2452 0.6717 deb:82, fin:84 p_values: 0.892 0.916
# allfolds valence: 0.2604 0.6759 arousal: 0.2455 0.6711 deb:84, fin:86 p_values: 0.981 0.915
# allfolds valence: 0.2607 0.6750 arousal: 0.2454 0.6712 deb:96, fin:98 p_values: 0.884 0.867
# allfolds valence: 0.2607 0.6750 arousal: 0.2454 0.6713 deb:130, fin:132 p_values: 0.976 0.984
# allfolds valence: 0.2605 0.6758 arousal: 0.2456 0.6706 deb:202, fin:204 p_values: 0.916 0.992
# allfolds valence: 0.2607 0.6750 arousal: 0.2456 0.6708 deb:204, fin:206 p_values: 0.871 0.888
# allfolds valence: 0.2605 0.6757 arousal: 0.2455 0.6708 deb:206, fin:208 p_values: 0.970 0.870
# allfolds valence: 0.2605 0.6756 arousal: 0.2453 0.6715 deb:220, fin:222 p_values: 0.970 0.925
# allfolds valence: 0.2606 0.6754 arousal: 0.2454 0.6713 deb:222, fin:224 p_values: 0.826 0.918
# allfolds valence: 0.2606 0.6751 arousal: 0.2453 0.6714 deb:228, fin:230 p_values: 0.887 0.999
#
# awk '($3<0.2608 && $6<0.2457){print}' retrait_baseline_features_ter.log
# allfolds valence: 0.2603 0.6760 arousal: 0.2455 0.6710 deb:76, fin:84 p_values: 0.622 0.697
# allfolds valence: 0.2604 0.6753 arousal: 0.2451 0.6722 deb:72, fin:84 p_values: 0.411 0.617
# allfolds valence: 0.2606 0.6764 arousal: 0.2455 0.6706 deb:84, fin:130 p_values: 0.015 0.025

# openSMILE_feat_indices_to_remove = [[14, 16], [72, 86], [96, 98], [130, 132], [202, 208], [220, 224], [228, 230]]
openSMILE_feat_indices_to_remove = [[14, 16], [72, 132], [202, 208], [220, 224], [228, 230]]
