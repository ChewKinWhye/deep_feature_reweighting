#! /bin/sh

for weight_decay in 1e-2 1e-3 1e-4
do
	for batch_size in 32 64 128
	do
		for init_lr in 1e-3 1e-4
		do
			for method_scale in 0.5 1 2
			do
				exp_name=$weight_decay-$batch_size-$init_lr-$method_scale
				echo $exp_name
				qsub -v weight_decay=$weight_decay,batch_size=$batch_size,init_lr=$init_lr,method_scale=$method_scale submit.sh
			done
		done
	done
done
