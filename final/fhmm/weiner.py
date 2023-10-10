# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:19:42 2022

@author: David
"""


import numpy as np

class inputParams():
    def __init__(self, means, stateSequence):
        self.m = means
        self.ss = stateSequence

class weinerFilter():
    """
    Weiner filter to separate audio

    Attributes
    ----------
    m1 : features x dimension matrix
        means of hmm object 01

    m2 : features x dimension matrix
        means of hmm object 02

    """
    def __init__(self, d):
        self.d = d

    """
    features: length x dimension of features
    ss01: state sequence for hmm01
    ss02: state sequence for hmm02
    """
    def getHardMask(self, features, model01: inputParams, model02: inputParams):
        frames = features.shape[0]
        mask01 = np.zeros(features.shape)
        mask02 = np.zeros(features.shape)
        for i in range(frames):
            mask01_means = model01.m[model01.ss[i]]
            mask02_means = model02.m[model02.ss[i]]
            mask_test = (mask01_means > mask02_means).astype(int)
            mask01[i, :] = mask_test
            mask02[i, :] = ((mask_test - 1) * -1)
        return mask01, mask02

    def getSoftMask(self, features, model01: inputParams, model02: inputParams):
        frames = features.shape[0]
        mask01 = np.zeros(features.shape)
        mask02 = np.zeros(features.shape)
        for i in range(frames):
            mask01_means = model01.m[int(model01.ss[i])]
            mask02_means = model02.m[int(model02.ss[i])]
            mask01[i, :] = mask01_means / (mask02_means + mask01_means)
            mask02[i, :] = mask02_means / (mask02_means + mask01_means)
        return mask01, mask02
