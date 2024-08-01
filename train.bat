@echo off
setlocal

:: Set the Python executable
set PYTHON_EXE=python

:: Set script name
set SCRIPT_NAME=train_model.py

:: Set parameters for the script
set MODEL_TYPE=xlnet
set FREEZE_LAYER=3
set FREEZE_EMBEDDING=True
set FREEZE_POOL=True
set FREEZE_SUMMARY=False
set INPUT_SIZE=768
set HIDDEN_LAYERS=128 64 32
set NUM_CLASSES=8
set DATASET_FOLDER=USs/user_stories_score_full.csv
set TRAIN_BATCH_SIZE=32
set TEST_BATCH_SIZE=32
set EPOCHS=5
set LEARNING_RATE=0.00002
set WEIGHT_DECAY=0.01
set LOG_FREQ=10
set DEVICE=cuda
set OUTPUT_FOLDER=output/xlnet

:: Run the Python script with the parameters
"%PYTHON_EXE%" "%SCRIPT_NAME%" ^
    --model_type %MODEL_TYPE% ^
    --freeze_layer %FREEZE_LAYER% ^
    --freeze_embedding %FREEZE_EMBEDDING% ^
    --freeze_pool %FREEZE_POOL% ^
    --freeze_summary %FREEZE_SUMMARY% ^
    --input_size %INPUT_SIZE% ^
    --hidden_layers %HIDDEN_LAYERS% ^
    --num_classes %NUM_CLASSES% ^
    --dataset_folder %DATASET_FOLDER% ^
    --train_batch_size %TRAIN_BATCH_SIZE% ^
    --test_batch_size %TEST_BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --lr %LEARNING_RATE% ^
    --weight_decay %WEIGHT_DECAY% ^
    --log_freq %LOG_FREQ% ^
    --device %DEVICE% ^
    --output_folder %OUTPUT_FOLDER%

:: Pause the script to view the output
pause
