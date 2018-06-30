#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:09:21 2018

@author: ajoshi
"""

#||AUM||
#||Shree Ganeshaya Namaha||
import fMRIlearn.cog_predictor as cpred
#import fMRIlearn.read_gord_data as rh

# def main():
""" Main script that calls the functions object"""
bfp_dir = '/home/ajoshi/coding_ground/bfp'
data_dir = '/deneb_disk/temp1'
csv_file = '/deneb_disk/temp1/Peking_all_phenotypic.csv'

cp = cpred.CogPred(bfp_dir)

nn = cp.get_neural_net()

cp.read_fmri(data_dir, reduce_dim=21)
cp.read_cog_scores(csv_file)

d, _, _ = cp.get_data()
l = cp.map_gord2sqrs()

print(l)
print(cp.nn_ipdata)
#    hcprfMRI = read_hcp_rfMRI(HCPpath)
#    hcpCogscores = read_hcp_cogscores(xlspath)
#   cp1.fit(gordfiledir)
#    cp.predict('hello')

# if __name__ == "__main__":
#    main()
