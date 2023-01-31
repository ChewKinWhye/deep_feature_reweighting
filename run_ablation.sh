#! /bin/sh

for method in 0 7
do
	for weight_decay in 1e-4
	do
		for batch_size in 16
		do
			for init_lr in 1e-4
			do
				for seed in 0
				do
					exp_name=$seed-$weight_decay-$batch_size-$init_lr
					echo $exp_name
					qsub -v seed=$seed,weight_decay=$weight_decay,batch_size=$batch_size,init_lr=$init_lr,method=$method submit.sh
				done
			done
		done
	done
done
