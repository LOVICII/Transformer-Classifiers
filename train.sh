#!/bin/bash

# Set the Python executable
PYTHON_EXE=python3

# Set script name
SCRIPT_NAME=train_model.py

# Set parameters for the script
MODEL_TYPE="bert"
FREEZE_LAYER=8
FREEZE_EMBEDDING="True"
FREEZE_POOL="True"
FREEZE_SUMMARY="True"
INPUT_SIZE=768
HIDDEN_LAYERS="256 128 64 32" 
NUM_CLASSES=8
DATASET_FOLDER="USs/user_stories_score_full.csv"
TRAIN_BATCH_SIZE=32
TEST_BATCH_SIZE=32
EPOCHS=2
LEARNING_RATE=0.00002
WEIGHT_DECAY=0.01
LOG_FREQ=10
DEVICE="cpu"
OUTPUT_FOLDER="output/roberta"

# Run the Python script with the parameters
$PYTHON_EXE $SCRIPT_NAME \
    --model_type $MODEL_TYPE \
    --freeze_layer $FREEZE_LAYER \
    --freeze_embedding $FREEZE_EMBEDDING \
    --freeze_pool $FREEZE_POOL \
    --freeze_summary $FREEZE_SUMMARY \
    --input_size $INPUT_SIZE \
    --hidden_layers $HIDDEN_LAYERS \
    --num_classes $NUM_CLASSES \
    --dataset_folder $DATASET_FOLDER \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --test_batch_size $TEST_BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --log_freq $LOG_FREQ \
    --device $DEVICE \
    --output_folder $OUTPUT_FOLDER
