#!/bin/bash

# Methods below uses the optimal hyperparameters found

CUDA_VISIBLE_DEVICES=3, python train.py --output_dir waterbird_erm --num_hparams_seeds 1 --num_init_seeds 5 --method erm --dataset waterbirds --lr 1e-4 --weight_decay 1e-3 --num_epochs 200 --batch_size 4
CUDA_VISIBLE_DEVICES=0, python train.py --output_dir waterbird_rwg --num_hparams_seeds 1 --num_init_seeds 5 --method rwg --dataset waterbirds --lr 1e-5 --weight_decay 1e-1 --num_epochs 90 --batch_size 36
CUDA_VISIBLE_DEVICES=1, python train.py --output_dir waterbird_subg --num_hparams_seeds 1 --num_init_seeds 5 --method subg --dataset waterbirds --lr 1e-4 --weight_decay 1e-3 --num_epochs 200 --batch_size 4
CUDA_VISIBLE_DEVICES=2, python train.py --output_dir waterbird_gDRO --num_hparams_seeds 1 --num_init_seeds 5 --method dro --dataset waterbirds --lr 1e-5 --weight_decay 1e-1 --num_epochs 20 --batch_size 4



# Base Model
CUDA_VISIBLE_DEVICES=0, python3 train_classifier.py --pretrained_model --augment_data \
  --weight_decay=1e-3 --batch_size=4 --init_lr=1e-4 --num_epochs 200 --output_dir=temp


###################################################################################
# Base Model
CUDA_VISIBLE_DEVICES=3, python3 train_classifier.py --output_dir=base_l2_reweight_y --pretrained_model \
  --num_epochs=100 --weight_decay=1e-3 --batch_size=64 --init_lr=1e-3 \
  --eval_freq=1 --augment_data --weight_decay_type l2 --reweight_classes
CUDA_VISIBLE_DEVICES=3, python3 dfr_evaluate_spurious.py \
  --result_path=feature_reg.pkl --ckpt_path=iteration-0.5-0/best_checkpoint.pt

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



for lr in 1e-5 1e-4 1e-3
do
    for weight_decay in 1e-4 1e-3 1e-2 1e-1 1
    do
        for batch_size in 64 128 256
        do
            for d_lr in 1e-5 1e-4 1e-3
            do
                for i_weight in 100 500 1000
                do
                    for seed in 0 1
                    do
                        CUDA_VISIBLE_DEVICES=0, python3 train_classifier.py --seed $seed --output_dir=$lr-$weight_decay-$batch_size-$d_lr-$i_weight-$seed --pretrained_model --num_epochs=100 --weight_decay=$weight_decay --batch_size=$batch_size --init_lr=$lr --eval_freq=1 --augment_data --weight_decay_type l2 --reweight_classes --conditional_independence --discriminator_lr $d_lr --independence_weight $i_weight
