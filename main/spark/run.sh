#!/usr/bin/env bash


# parameters to be set
MASTER = $SPARKURL
NUMEXECUTORS = 63
NUMCORES = 1
DRIVERMEMORY = 10G
EXECUTORMEMORY = 2G
PYTHONFILE = $SCRATCH/SketchedRidgeRegression/main/spark/modelAvgTxtFile.py


spark-submit $PYTHONFILE \
  --verbose \
  --master $MASTER \
  --num-executors $NUMEXECUTORS \
  --executor-cores $NUMCORES \
  --driver-memory $DRIVERMEMORY \
  --executor-memory $EXECUTORMEMORY
