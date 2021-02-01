#!/bin/bash

python main.py \
--name monodepth \
--isTrain \
--batchsize 16 \
--dataroot '../../dataset/kitti/data/' \
--filename './data/filenames/kitti_train_files.txt'