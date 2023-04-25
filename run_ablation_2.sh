#! /bin/sh

method=2
dataset="mcdominoes"
regularize_mode=3

weight_decay=1e-3
batch_size=16
init_lr=1e-3
group_size=8

for spurious_strength in 0.8 0.9 0.95 1
do
	for val_target_size in 1000 2000 6000
	do
		for seed in 0 1 2
		do
			exp_name=$method-$regularize_mode-$dataset-$spurious_strength-$val_size-$weight_decay-$batch_size-$group_size-$init_lr-$seed
        		echo $exp_name
			qsub -v method=$method,regularize_mode=$regularize_mode,dataset=$dataset,spurious_strength=$spurious_strength,val_target_size=$val_target_size,weight_decay=$weight_decay,batch_size=$batch_size,init_lr=$init_lr,group_size=$group_size,seed=$seed submit_2.sh
		done
	done
done
