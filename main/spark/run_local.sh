#!/usr/bin/env bash

export INPUT_DIR="$HOME/Code/SketchedRidgeRegression"
export NUM_EXECUTORS="678"

SPARK_HOME="$HOME/Software/spark-2.0.2"
MASTER="local"
PYTHON_FILE="$INPUT_DIR/main/spark/modelAvg.py"



$SPARK_HOME/bin/spark-submit $PYTHON_FILE \
  --verbose \
  --master $MASTER 
