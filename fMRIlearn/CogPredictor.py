#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:16:45 2018

@author: ajoshi
"""
import numpy as np
import scipy as sp

# Define cognitive predictor regressor
# This takes fMRI grayordinate data as input and cognitive scores as output
class CogPredictor:

    def __init__(self, bfp_path):
        print("This is the constructor method.")
        # read the flat maps for both hemispheres
        sqr_left = readdfs(bfp_path)
        sqr_right = readdfs()

    def map_gord2sqrs(self, gord_data, sqr_size):
        X, Y = sp.meshgrid(range(sqr_size), range(sqr_size))
        left_data = 
        interp

    def fit(self, X, y):
        # X: data in grayordinates of shape Vert x Time x Subj
        # y: cognitive scores
        print('Fitting the model')
        u_net = get_nn()

    def predict(str1):
        print(str1)

    def get_NN():
