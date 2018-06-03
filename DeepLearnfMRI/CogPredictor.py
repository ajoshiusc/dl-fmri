#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:16:45 2018

@author: ajoshi
"""


# Define cognitive predictor regressor
# This takes fMRI grayordinate data as input and cognitive scores as output
class CogPredictor:

    def __init__(self, bfp_path):
        print("This is the constructor method.")
        # read the flat maps for both hemispheres
        sqr_left = readdfs(bfp_path)
        sqr_right = readdfs()

    def fit(self, X, y):
        print(str11)

    def predict(str1):
        print(str1)