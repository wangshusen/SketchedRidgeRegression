#!/usr/bin/env bash

wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2
bzip2 -d YearPredictionMSD.bz2
bzip2 -d YearPredictionMSD.t.bz2
