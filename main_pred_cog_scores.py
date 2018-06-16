#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:09:21 2018

@author: ajoshi
"""

#||AUM||
#||Shree Ganeshaya Namaha||
import fMRIlearn.cog_predictor as cp
import fMRIlearn.read_gord_data as rh

def main():
    """ Main script that calls the functions object"""
    bfp_path = '/home/ajoshi/coding_ground/bfp'
    data_dir = '/deneb_disk/ADHD_Peking_bfp'
    xlspath = ''
    cp1 = cp.CogPred(bfp_path)
    hpc = rh.bfpData(data_dir)
    hpc.read_fMRI()

#    hcprfMRI = read_hcp_rfMRI(HCPpath)
#    hcpCogscores = read_hcp_cogscores(xlspath)
#   cp1.fit(gordfiledir)
#    cp.predict('hello')

if __name__ == "__main__":
    main()
