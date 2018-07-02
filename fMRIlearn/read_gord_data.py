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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from fMRIlearn.brainsync import brainSync, normalizeData


class bfpData():
    """ This class manages BFP data set"""

    def __init__(self):
        # initialization
        self.rfmri = 0
        self.cog_scores = 0
        self.nvert_hemi = 32492
        self.nvert_subcort = 31870
        self.data_dir = ""
        self.subids = list()
        self.data = list()
        self.dirlst = list()
        print("Read flat maps for left and right hemispheres.")

    def get_data(self):
        """get the rfMRI data"""
        return self.data, self.cog_scores, self.subids
    
    def choose_rep_sub(self):
        self.rep_subno = 0



    def read_fmri(self, data_dir, reduce_dim=None, int_subid=1):
        """ Read fMRI data from disk """
        """ If reduce_dim = None, no dimesionality reduction is performed"""
        self.data_dir = data_dir
        self.dirlst = glob.glob(self.data_dir+'/*.mat')

        if reduce_dim != None :
            pca = PCA(n_components=reduce_dim)

        for subfile in self.dirlst:
            subid = subfile.replace('_rest_bold.32k.GOrd.mat', '')
            subid = subid.replace(self.data_dir + '/', '')
            if int_subid:
                subid = int(subid)

            if os.path.isfile(subfile):
                print('Reading '+ subfile, 'subid = ' + str(subid))
                fmri_data = loadmat(subfile)['dtseries']
                fmri_data, _, _ = normalizeData(fmri_data.T)
                fmri_data = fmri_data.T
                
                if reduce_dim != None:
                    fmri_data = pca.fit_transform(fmri_data)

                self.subids.append(subid)
                self.data.append(fmri_data)
#               print(subid, subfile)



    def read_cog_scores(self, cogscore_file):
        """ Read cognitive scores from csv file """
        self.cog_scores = pd.read_csv(cogscore_file, index_col=0)

        ''' If fMRI data exists for some subjects, then store their cognitive scores ''' 
        for subid in self.subids:
            self.cog_scores.append(self.get_cog_score_subid(subid))

        print(self.cog_scores)

    def get_cog_score_subid(self,subid):
        """ Get cognitive score for a given subject id"""
        return self.cog_scores.loc[subid]

