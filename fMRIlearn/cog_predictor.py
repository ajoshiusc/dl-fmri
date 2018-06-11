#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:16:45 2018

@author: ajoshi
"""
import os
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import griddata
from sklearn.base import BaseEstimator


# Define cognitive predictor regressor This takes fMRI grayordinate data as
# input and cognitive scores as output


# This takes fMRI grayordinate data as input and cognitive scores as output
class CogPred(BaseEstimator):
    sqrmap = 0

    def __init__(self, bfp_dir):
        print("Reading flat maps for left and right hemispheres.")
        # read the flat maps for both hemispheres
        dat = loadmat(os.path.join(bfp_dir))
        self.sqrmap = dat['sqrmap']

    def map_gord2sqrs(self, data, sqr_size=256):
        """This function maps grayordinate data to square
        flat map flat map of 32k vertices,
        data: data defined on 32k vertices,
        sqr_size: size of the square
        """

        x_ind, y_ind = np.meshgrid(np.linspace(-1.0, 1.0, sqr_size),
                                   np.linspace(-1.0, 1.0, sqr_size))

        sqr_data = np.zeros((x_ind.shape(0), y_ind.shape(1), data.shape[1]))

        for t_ind in np.arange(data.shape[1]):
            sqr_data[:, :, t_ind] = griddata(self.sqrmap, data[:, t_ind], (x_ind, y_ind))

        return sqr_data

    def fit(self, X, y):
        """ X: data in grayordinates of shape Vert x Time x Subj
            y: cognitive scores"""
        map_gord2sqrs(X)
        print('Fitting the model')
        u_net = get_nn()

    def predict(self, str1):
        print(str1)

    def get_NN():
        pass
# fdfdh
