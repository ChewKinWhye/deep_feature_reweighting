CUDA_VISIBLE_DEVICES=3, python3 train_classifier.py --seed 0 --pretrained_model --augment_data --weight_decay=1e-1 \
--batch_size=32 --init_lr=1e-4 --num_epochs 100 --output_dir=COND_CORAL-0 --method 5 --method_scale 0.5 --reweight_classes
