'''
MODEL    :: Pytorch Computer Vision Platform
DATE    :: 2021-01-26
FILE     :: main.py 
'''

from __future__ import absolute_import, division, print_function

from config.options import Options
from engine.train import Train

opt = Options().parse() 

if __name__ == "__main__":
    if opt.isTrain:
        trainer = Train(opt).train()             
    else:
        print("--- Wrong Mode ---")
