#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:09:21 2018

@author: ajoshi
"""

#||AUM||
#||Shree Ganeshaya Namaha||
from DeepLearnfMRI import CogPredictor as cp

def main():
    # Set name of Shark object
    cp1 = cp.CogPredictor()
    cp1.fit('hi11')
#    cp.predict('hello')

if __name__ == "__main__":
    main()
