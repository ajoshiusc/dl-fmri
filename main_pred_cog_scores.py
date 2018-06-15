#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:09:21 2018

@author: ajoshi
"""

#||AUM||
#||Shree Ganeshaya Namaha||
import fMRIlearn.cog_predictor as cp

def main():
    """ Main script that calls the functions object"""
    bfp_path = '/home/ajoshi/coding_ground/bfp'
    cp1 = cp.CogPred(bfp_path)

    hcprfMRI = read_HCP_rfMRI(HCPpath)
    hcpCogscores = read_HCP_cogscores(xlspath)
#   cp1.fit(gordfiledir)
#    cp.predict('hello')

if __name__ == "__main__":
    main()
