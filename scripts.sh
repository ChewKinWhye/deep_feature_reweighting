# For Baseline ERM 87.1%
#for seed in 0 1 2 3 4
#do
#    CUDA_VISIBLE_DEVICES=3, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=1e-3 \
#    --batch_size=4 --init_lr=1e-4 --num_epochs 200 --output_dir=ERM-$seed
#done

# For RWG
# for seed in 0 1 2 3 4
# do
#     CUDA_VISIBLE_DEVICES=3, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=1e-1 \
#     --batch_size=36 --init_lr=1e-5 --num_epochs 200 --output_dir=RWG-$seed --reweight_groups
# done

# For RWY
#for seed in 0 1 2 3 4
#do
#    CUDA_VISIBLE_DEVICES=3, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=1e-2 \
#    --batch_size=36 --init_lr=1e-4 --num_epochs 200 --output_dir=RWY-$seed --reweight_classes
#done

# For CnC + RWG
for seed in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=3, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=1e-1 --batch_size=36 --init_lr=1e-5 \
    --num_epochs 20 --output_dir=contrast-all-$seed --feature_reg_type contrast --feature_reg 3 \
    --reweight_groups --contrast_temperature 0.07
done

# CnC Half + RWG
for seed in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=2, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=1e-1 --batch_size=36 --init_lr=1e-5 \
    --num_epochs 20 --output_dir=contrast_half-5-0.1-$seed --feature_reg_type contrast --feature_reg 5 \
    --reweight_groups --contrast_temperature 0.1 --inverse_contrast

done

# For CnC Tenth + RWG
for seed in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=3, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=1e-1 --batch_size=36 --init_lr=1e-5 \
    --num_epochs 20 --output_dir=contrast_tenth-5-0.07-$seed --feature_reg_type contrast --feature_reg 5 \
    --reweight_groups --contrast_temperature 0.07 --inverse_contrast --contrast_tenth
done

# For CORAL
for seed in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=2, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=1e-1 \
    --batch_size=32 --init_lr=1e-4 --num_epochs 100 --output_dir=CORAL-1e-1-0.5-$seed --coral --feature_reg 0.5
done