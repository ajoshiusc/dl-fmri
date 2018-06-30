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


class CogPred(BaseEstimator):
    '''This takes fMRI grayordinate data as input and cognitive scores as output'''

    def __init__(self, bfp_dir):
        # read the flat maps for both hemispheres
        dat = loadmat(os.path.join(bfp_dir, 'supp_data', 'sqrmap.mat'))
        self.sqrmap = dat['sqrmap']
        # for converting indices from matlab to python, subtract -1
        self.sqr_map_ind = dat['data_ind'] - 1
        self.sqr_map_ind = self.sqr_map_ind.squeeze()
        print("Read flat maps for left and right hemispheres.")
        self.nvert_hemi = 32492

    def map_gord2sqrs(self, data, sqr_size=256):
        """This function maps grayordinate data to square
        flat map flat map of 32k vertices,
        data: data defined on 32k vertices,
        sqr_size: size of the square
        """
        print(sqr_size)
        x_ind, y_ind = np.meshgrid(np.linspace(-1.0+1e-6, 1.0-1e-6, sqr_size),
                                   np.linspace(-1.0+1e-6, 1.0-1e-6, sqr_size))

        print(x_ind.shape)
        sqr_data_sz = (x_ind.shape[0], y_ind.shape[1], data.shape[1])
        sqr_data_left = np.zeros(sqr_data_sz)
        sqr_data_right = np.zeros(sqr_data_sz)

        for t_ind in np.arange(data.shape[1]):
            # Map Left Hemisphere data
            lh_data = data[:self.nvert_hemi, t_ind][self.sqr_map_ind]
            sqr_inds = (x_ind, y_ind)
            sqr_data_left[:, :, t_ind] = griddata(self.sqrmap, lh_data,
                                                  sqr_inds)
            # Map Right Hemisphere data
            rh_data = data[self.nvert_hemi:2*self.nvert_hemi, t_ind]
            rh_data = rh_data[self.sqr_map_ind]
            sqr_data_right[:, :, t_ind] = griddata(self.sqrmap,
                                                   rh_data, sqr_inds)
            print(str(t_ind) + ',', end='', flush=True)

        noncortical_data = data[2*self.nvert_hemi:, ]
        return sqr_data_left, sqr_data_right, noncortical_data

    def fit(self, X, y):
        """ X: data in grayordinates of shape Vert x Time x Subj
            y: cognitive scores"""
        self.map_gord2sqrs(X)
        print('Fitting the model')
        u_net = self.get_neural_net()

    def predict(self, str1):
        print(str1)

    def get_neural_net(self):
        
        pass

   

# fdfdh

