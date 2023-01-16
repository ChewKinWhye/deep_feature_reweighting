# For Conditional Coral
for seed in 0 1 2
do
    CUDA_VISIBLE_DEVICES=2, python3 train_classifier.py --seed $seed --pretrained_model --augment_data --weight_decay=1e-3 \
    --batch_size=32 --init_lr=1e-4 --num_epochs 100 --output_dir=COND-CORAL-$seed --method 5 --method_scale 0.5 --reweight_classes
done


python3 tune_classifier.py --seed 0 --pretrained_model --augment_data \
--weight_decay=1e-3 \
--batch_size=32 --init_lr=1e-3 \
--num_epochs 100 --output_dir=temp --method 5 --method_scale 0.5 \
--reweight_classes
