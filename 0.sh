# For Contrast Tenth
for lr in 1e-3 1e-4
do
    for batch_size in 32 64 128
    do
        for weight_decay in 1e-2 1e-3
        do
            for temperature in 0.07 0.1 0.7
            do
                for method_scale in 0.1 0.5 1
                do
                    for seed in 0 1 2
                    do
                    done
                done
            done
        done
    done
done
CUDA_VISIBLE_DEVICES=3, python3 tune_classifier.py --seed 0 --pretrained_model --augment_data \
--weight_decay=1e-3 \
--batch_size=32 --init_lr=1e-3 \
--num_epochs 100 --output_dir=contrast_tenth-0 --method 2 --method_scale 0.5 \
--reweight_classes --contrast_temperature 0.07
