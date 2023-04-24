#! /bin/sh

for method in 0
do
	for dataset in mcdominoes
	do
		for spurious_strength in 1 
		do
			for val_size in 1000
			do
				for weight_decay in 1e-1 1e-2 1e-3 1e-4
				do
					for batch_size in 4 8 16 32
					do
						for init_lr in 1e-2 1e-3 1e-4
						do
							for seed in 0 1 2
							do
								exp_name=$method-$dataset-$spurious_strength-$val_size-$weight_decay-$batch_size-$init_lr-$seed
								echo $exp_name
								qsub -v method=$method,dataset=$dataset,spurious_strength=$spurious_strength,val_size=$val_size,weight_decay=$weight_decay,batch_size=$batch_size,init_lr=$init_lr,seed=$seed submit_0.sh
							done
						done
					done
				done
			done
		done
	done
done
