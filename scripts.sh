# Method 0: Normal ERM
python train_classifier.py --pretrained_model --augment_data --num_epochs 100 --batch_size 64  --weight_decay 1e-4 --init_lr 1e-3 --reweight_classes --method 0

# Method 5: Conditional CORAL
python train_classifier.py --pretrained_model --augment_data --num_epochs 100 --batch_size 32  --weight_decay 1e-4 --init_lr 1e-3 --reweight_classes --method 5

# Method 7: MTL
python train_classifier.py --pretrained_model --augment_data --num_epochs 100 --batch_size 64  --weight_decay 1e-4 --init_lr 1e-3 --reweight_classes --method 7

