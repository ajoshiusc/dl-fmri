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
        dat = loadmat(os.path.join(bfp_dir,'supp_data','sqrmap.mat'))
        self.sqrmap = dat['sqrmap']
        self.sqr_map_ind = dat['data_ind']
        print("Read flat maps for left and right hemispheres.")


    def map_gord2sqrs(self, data, sqr_size=256):
        """This function maps grayordinate data to square
        flat map flat map of 32k vertices,
        data: data defined on 32k vertices,
        sqr_size: size of the square
        """
        print(sqr_size)
        x_ind, y_ind = np.meshgrid(np.linspace(-1.0, 1.0, sqr_size),
                                   np.linspace(-1.0, 1.0, sqr_size))

        print(x_ind.shape)
        sqr_data_left = np.zeros((x_ind.shape[0], y_ind.shape[1], data.shape[1]))
        sqr_data_right = np.zeros((x_ind.shape[0], y_ind.shape[1], data.shape[1]))

        nvert_hemi = self.sqrmap.shape[0]
        for t_ind in np.arange(data.shape[1]):
            lh_data = data[:nvert_hemi, t_ind]
            sqr_data_left[:, :, t_ind] = griddata(self.sqrmap, lh_data, (x_ind, y_ind))
            rh_data = data[nvert_hemi:2*nvert_hemi, t_ind]
            sqr_data_right[:, :, t_ind] = griddata(self.sqrmap, rh_data, (x_ind, y_ind))
           
            print(str(t_ind) + ',', end='')

        noncortical_data = data[2*nvert_hemi:, ]
        return sqr_data_left, sqr_data_right, noncortical_data
    
    def fit(self, X, y):
        """ X: data in grayordinates of shape Vert x Time x Subj
            y: cognitive scores"""
        self.map_gord2sqrs(X)
        print('Fitting the model')
        u_net = get_nn()

    def predict(self, str1):
        print(str1)

    def get_neural_net(self):
        pass

   

# fdfdh

