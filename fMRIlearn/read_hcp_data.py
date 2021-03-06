#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:16:45 2018

@author: ajoshi
"""

import nilearn as nl
import os
import scipy as sp
from scipy.io import loadmat


class HCPData():
    """ This class manages HCP data"""

    def __init__(self, hcp_dir):
        # initialization
        self.rfmri = 0
        self.cog_scores = 0
        self.hcp_dir = hcp_dir
        self.subids = list()
        print("Read flat maps for left and right hemispheres.")

    def get_data(self):
        """get the rfMRI data"""
        self.read_fMRI()
    #    self.read_cog_scores(self.subids)
        return self.rfmri, self.cog_scores

    def read_fMRI(self):
        """ Read fMRI data from disk """
        dirlst = os.listdir(self.hcp_dir)

        for subid in dirlst:
            fname = os.path.join(self.hcp_dir, subid, 'MNINonLinear',
                                 'Results', 'rfMRI_REST1_LR',
                                 'rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.mat')
            if os.path.isfile(fname):
                dat = loadmat(fname)
                self.subids.append(subid)
                print(self.subids)
