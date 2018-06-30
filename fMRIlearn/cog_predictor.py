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
from fMRIlearn.read_gord_data import bfpData
from keras.layers import Input,Conv2D,concatenate,MaxPooling2D,Flatten,Dense,Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import losses

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


# Define cognitive predictor regressor This takes fMRI grayordinate data as
# input and cognitive scores as output


class CogPred(BaseEstimator, bfpData):
    '''This takes fMRI grayordinate data as input and cognitive scores as output'''

    def __init__(self, bfp_dir):
        # read the flat maps for both hemispheres
        bfpData.__init__(self)
        dat = loadmat(os.path.join(bfp_dir, 'supp_data', 'sqrmap.mat'))
        self.sqrmap = dat['sqrmap']
        # for converting indices from matlab to python, subtract -1
        self.sqr_map_ind = dat['data_ind'] - 1
        self.sqr_map_ind = self.sqr_map_ind.squeeze()
        self.sqr_data = list()
        self.nn_ipdata = list()
        print("Read flat maps for left and right hemispheres.")
        self.nvert_hemi = 32492

    def map_gord2sqrs(self, sqr_size=256):
        """This function maps grayordinate data to square
        flat map flat map of 32k vertices,
        data: data defined on 32k vertices,
        sqr_size: size of the square
        """
        print(sqr_size)
        x_ind, y_ind = np.meshgrid(np.linspace(-1.0+1e-6, 1.0-1e-6, sqr_size),
                                   np.linspace(-1.0+1e-6, 1.0-1e-6, sqr_size))

        print(x_ind.shape)
        sqr_data_sz = (x_ind.shape[0], y_ind.shape[1], self.data[0].shape[1])
        sqr_data_left = np.zeros(sqr_data_sz)
        sqr_data_right = np.zeros(sqr_data_sz)

        for data in self.data:
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

            noncortical_data = np.nan_to_num(data[2*self.nvert_hemi:, ])
            sqr_data_left = np.nan_to_num(sqr_data_left)
            sqr_data_right = np.nan_to_num(sqr_data_right)

            def sqr_dat():
                pass

            sqr_dat.sqr_data_left = sqr_data_left
            sqr_dat.sqr_data_right = sqr_data_right
            sqr_dat.noncortical_data = noncortical_data
            self.nn_ipdata.append(sqr_dat)

        return self.nn_ipdata

    def fit(self, X, y):
        """ X: data in grayordinates of shape Vert x Time x Subj
            y: cognitive scores"""
        self.map_gord2sqrs(X)
        print('Fitting the model')
        u_net = get_neural_net()

    def predict(self, str1):
        print(str1)

    def get_neural_net(self, isize=[256, 256]):
        input_lh = Input((isize[0], isize[1], 21))
        input_rh = Input((isize[0], isize[1], 21))
        sub_cort = Input((32492, 21))

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_lh)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_rh)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        d1_ip = Dense(512,activation='relu')(sub_cort)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4_1)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        flat1 = Flatten()(conv5)
        d1= Dense(512,activation='relu')(flat1, d1_ip)
        d2= Dense(64,activation='relu')(d1)

        out_theta = Dense(1)(d2)
    #    conv_tx = Conv2D(1, (1, 1), activation=final_activation)(conv5)
    #    conv_ty = Conv2D(1, (1, 1), activation=final_activation)(conv5)
    #    conv_theta = Conv2D(1, (1, 1), activation='tanh')(conv5)

    #    out_img = rotate(inputs,conv_theta)

        model = Model(inputs=[input_lh, input_rh, sub_cort], outputs=out_theta)

        model.compile(optimizer='adam', loss=losses.mean_squared_error, metrics=['mse'])

        return model


# fdfdh
