#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:16:45 2018

@author: ajoshi
"""
import numpy as np
import scipy as sp
from scipy.io import loadmat
from sklearn.base import BaseEstimator
from scipy.interpolate import griddata
import os


# Define cognitive predictor regressor
# This takes fMRI grayordinate data as input and cognitive scores as output
class CogPred(BaseEstimator):
    sqrmap = 0

    def __init__(self, bfp_dir):
        print("Reading flat maps for left and right hemispheres.")
        # read the flat maps for both hemispheres
        dat = loadmat(os.path.join(bfp_dir))
        self.sqrmap = dat['sqrmap']

# This function maps grayordinate data to square
# flat map flat map of 32k vertices, data: data defined on 32k vertices
# sqr_size: size of the square
    def map_gord2sqrs(self, data, sqr_size):
        X, Y = sp.meshgrid(np.linspace(-1.0, 1.0, sqr_size),
                           np.linspace(-1.0, 1.0, sqr_size))

        sqr_data = np.zeros((X.shape(0), X.shape(1), gord_data.shape[1]))

        for t in np.arange(data.shape[1]):
            sqr_data[:, :, t] = griddata(self.sqrmap, data[:, t], (X, Y))

        return sqr_data

    def fit(self, X, y):
        # X: data in grayordinates of shape Vert x Time x Subj
        # y: cognitive scores
        map_gord2sqrs()
        print('Fitting the model')
        u_net = get_nn()

    def predict(str1):
        print(str1)

    def get_NN():
        pass
# fdfdh