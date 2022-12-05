#!/bin/bash

###################################################################################
# Base Model
CUDA_VISIBLE_DEVICES=2, python3 train_classifier.py --output_dir=base_l2_reweight_y --pretrained_model \
  --num_epochs=100 --weight_decay=1e-3 --batch_size=64 --init_lr=1e-3 \
  --eval_freq=1 --augment_data --weight_decay_type l2 --reweight_classes
CUDA_VISIBLE_DEVICES=2, python3 dfr_evaluate_spurious.py \
  --result_path=feature_reg.pkl --ckpt_path=base_l2/final_checkpoint.pt \
  --tune_class_weights_dfr_train

## Base Model with L1 Norm
CUDA_VISIBLE_DEVICES=1, python3 train_classifier.py --output_dir=base_l1 --pretrained_model \
  --num_epochs=100 --weight_decay=1e-3 --batch_size=32 --init_lr=1e-3 \
  --eval_freq=1 --augment_data --weight_decay_type l1

## Base Model with L1 Norm with increasing decay
CUDA_VISIBLE_DEVICES=2, python3 train_classifier.py --output_dir=base_l1 --pretrained_model \
  --num_epochs=100 --weight_decay=1e-3 --batch_size=32 --init_lr=1e-3 \
  --eval_freq=1 --augment_data --weight_decay_type l1 --increasing_decay

# Multitask Model
CUDA_VISIBLE_DEVICES=1, python3 train_classifier.py --output_dir=multitask --pretrained_model \
  --num_epochs=100 --weight_decay=1e-3 --batch_size=32 --init_lr=1e-3 \
  --eval_freq=1 --augment_data --multitask --weight_decay_type l2 &
CUDA_VISIBLE_DEVICES=2, python3 dfr_evaluate_spurious.py \
  --result_path=feature_reg.pkl --ckpt_path=multitask/tmp_checkpoint.pt \
  --tune_class_weights_dfr_train

# FeatureReg Specific Model
CUDA_VISIBLE_DEVICES=2, python3 train_classifier.py --output_dir=feature_reg_specific --pretrained_model \
  --num_epochs=100 --weight_decay=1e-3 --batch_size=32 --init_lr=1e-3 \
  --eval_freq=1 --augment_data --feature_reg_type specific --feature_reg 0.1 --reweight_groups --weight_decay_type l2 &
CUDA_VISIBLE_DEVICES=2, python3 dfr_evaluate_spurious.py \
  --result_path=feature_reg.pkl --ckpt_path=feature_reg_specific/tmp_checkpoint.pt \
  --tune_class_weights_dfr_train

# FeatureReg General Model
CUDA_VISIBLE_DEVICES=3, python3 train_classifier.py --output_dir=feature_reg_general --pretrained_model \
  --num_epochs=100 --weight_decay=1e-3 --batch_size=32 --init_lr=1e-3 \
  --eval_freq=1 --augment_data --feature_reg_type general --feature_reg 0.1 --reweight_groups --weight_decay_type l2 &
CUDA_VISIBLE_DEVICES=3, python3 dfr_evaluate_spurious.py \
  --result_path=feature_reg.pkl --ckpt_path=feature_reg_general/tmp_checkpoint.pt \
  --tune_class_weights_dfr_train

