#!/bin/bash
source scl_source enable devtoolset-8 rh-python36
set -e

cd ..

HARDWARE="$1"

if [ $HARDWARE == "cpu" ]; then
    python tf_cnn_benchmarks.py \
        --device=cpu --batch_size=32 --model=resnet50 \
        --variable_update=independent --data_format=NHWC \
        --nodistortions --weight_decay=1e-4 --optimizer=momentum \
        --gradient_repacking=8 --train_dir=/tf_cnn/train 
elif [ $HARDWARE == "gpu" ]; then
    python tf_cnn_benchmarks.py \
        --batch_size=10 --model=resnet50 \
        --optimizer=momentum --variable_update=independent \
        --nodistortions --gradient_repacking=8 --num_epochs=90 \
        --weight_decay=1e-4 --use_fp16 \
        --train_dir=/tf_cnn/train \
        --num_gpu=2 --data_format=NCHW
else
    echo "Must specify either 'cpu' or 'gpu'."
fi
