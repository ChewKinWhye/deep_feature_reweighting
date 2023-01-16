
# For Contrast All
for temperature in 0.07 0.1 0.4 0.7
do
    for feature_reg in 0.5 1 3 5
    do
        for seed in 0 1 2
        do
            CUDA_VISIBLE_DEVICES=3, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=1e-1 --batch_size=36 --init_lr=1e-5 \
            --num_epochs 20 --output_dir=contrast-all-$feature_reg-$temperature-$seed --feature_reg_type contrast --feature_reg $feature_reg \
            --reweight_groups --contrast_temperature $temperature
        done
    done
done

# For Contrast Half
for temperature in 0.07 0.1 0.4 0.7
do
    for feature_reg in 0.5 1 3 5
    do
        for seed in 0 1 2
        do
            CUDA_VISIBLE_DEVICES=2, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=1e-1 --batch_size=36 --init_lr=1e-5 \
            --num_epochs 20 --output_dir=contrast_half-$feature_reg-$temperature-$seed --feature_reg_type contrast --feature_reg $feature_reg \
            --reweight_groups --contrast_temperature $temperature --inverse_contrast
        done
    done
done

# For Contrast Tenth
for temperature in 0.07 0.1 0.4 0.7
do
    for feature_reg in 0.5 1 3 5
    do
        for seed in 0 1 2
        do
            CUDA_VISIBLE_DEVICES=3, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=1e-1 --batch_size=36 --init_lr=1e-5 \
            --num_epochs 20 --output_dir=contrast_tenth-$feature_reg-$temperature-$seed --feature_reg_type contrast --feature_reg $feature_reg \
            --reweight_groups --contrast_temperature $temperature --inverse_contrast --contrast_tenth
        done
    done
done


# For Coral
for feature_reg in 0.5 1 3 5
do
    for weight_decay in 1e-1 1e-2 1e-3 1e-4
    do
        for seed in 0 1 2
        do
            CUDA_VISIBLE_DEVICES=0, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=$weight_decay \
            --batch_size=32 --init_lr=1e-4 --num_epochs 100 --output_dir=CORAL-$weight_decay-$feature_reg-$seed --coral --feature_reg $feature_reg
        done
    done
done

# For conditional independence via correlation matrix
for feature_reg in 0.5 1 3 5
do
    for seed in 0 1 2
    do
        CUDA_VISIBLE_DEVICES=2, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=1e-2 \
        --batch_size=36 --init_lr=1e-4 --num_epochs 200 --output_dir=CORR-$feature_reg-$seed --reweight_classes --feature_reg $feature_reg \
        --conditional_independence_2
    done
done