#!/bin/bash
set -e
cd /opt/hotword/data
echo "[features] starting $(date)"
wget -q -c "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy" -O validation_set_features.npy
echo "[features] validation_set_features.npy done $(date)"
wget -q -c "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" -O openwakeword_features_ACAV100M_2000_hrs_16bit.npy
echo "[features] ACAV100M (17GB) done $(date)"
ls -lh /opt/hotword/data
echo "DL_FEATURES_DONE $(date)"
