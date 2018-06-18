#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:16:45 2018

@author: ajoshi
"""

import nilearn as nl
import os
import glob
import scipy as sp
from scipy.io import loadmat
import pandas as pd


class bfpData():
    """ This class manages HCP data"""

    def __init__(self):
        # initialization
        self.rfmri = 0
        self.cog_scores = 0
        self.data_dir = ""
        self.subids = list()
        print("Read flat maps for left and right hemispheres.")

    def get_data(self):
        """get the rfMRI data"""
        self.read_fMRI(self,self.data_dir)
        self.read_cog_scores(self.subids)
        return self.rfmri, self.cog_scores

    def read_fMRI(self, data_dir):
        """ Read fMRI data from disk """
        self.data_dir = data_dir
        dirlst = glob.glob(self.data_dir+'/*.mat')

        fmri_dat = np.zeros((96000,len(dirlist)))
        subno = 0
        for subfile in dirlst:
            subid = subfile.replace('_rest_bold.32k.GOrd.mat', '')
            subid = subid.replace(self.data_dir + '/', '')

            fname = os.path.join(self.data_dir, subfile)

            if os.path.isfile(fname):
                print('Reading '+ fname)
                fmri_dat[,subno++] = loadmat(fname)['ftdata']
                self.subids.append(subid)
                print(subid, subfile)

    def read_cog_scores(self, cogscore_file):
        """ Read cognitive scores from csv file """
        self.cog_scores = pd.read_csv(cogscore_file, index_col=0)
        print(self.cog_scores)

    def get_cog_score(self,subid):
        """ Get cognitive score for a given subject id"""
        return self.cog_scores.loc[subid]

