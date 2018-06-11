#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:16:45 2018

@author: ajoshi
"""
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator

# Define cognitive predictor regressor
# This takes fMRI grayordinate data as input and cognitive scores as output
class CogPredictor(BaseEstimator, bfp_path):

    def __init__(self, bfp_path):

        print("This is the constructor method.")
        # read the flat maps for both hemispheres
        self.sqr_left = readdfs(bfp_path)
        self.sqr_right = readdfs()

    def map_gord2sqrs(self, gord_data, sqr_size):
        X, Y = sp.meshgrid(np.linspace(-1.0, 1.0, sqr_size),
                           np.linspace(-1.0, 1.0, sqr_size))
        for t in range(gord_data.shape(1)):
            
        left_data = 
        interp

    def fit(self, X, y):
        # X: data in grayordinates of shape Vert x Time x Subj
        # y: cognitive scores
        map_gord2sqrs()
        print('Fitting the model')
        u_net = get_nn()

    def predict(str1):
        print(str1)

    def get_NN():
