#!/usr/bin/env bash

export INPUT_DIR="$SCRATCH/SketchedRidgeRegression"
export NUM_EXECUTORS="95"

MASTER="$SPARKURL"
PYTHON_FILE="$INPUT_DIR/main/spark/modelAvgTxtFile.py"



spark-submit $PYTHON_FILE \
  --verbose \
  --master $MASTER 
  --num-executors $NUM_EXECUTORS 

