# For CnC 1/100 + RWG
for seed in 0 1 2
do
    CUDA_VISIBLE_DEVICES=1, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=1e-3 --batch_size=32 --init_lr=1e-4 \
    --num_epochs 100 --output_dir=contrast_hundredth-$seed --method 3 --method_scale 0.5 \
    --reweight_classes --contrast_temperature 0.07
done
