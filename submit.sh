#! /bin/sh

#PBS -q volta_gpu
#PBS -j oe
#PBS -N pytorch
#PBS -l select=1:ncpus=10:mem=80gb:ngpus=1
#PBS -l walltime=8:00:00
#PBS -P 11002407

cd $PBS_O_WORKDIR;
np=$(cat ${PBS_NODEFILE} | wc -l);
#image="/app1/common/singularity-img/3.0.0/pytorch_1.9_cuda_11.3.0-ubuntu20.04-py38-ngc_21.04.simg"
image="/app1/common/singularity-img/3.0.0/pytorch_1.11_cuda_11.3_cudnn8-py38.sif"


singularity exec $image bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID
export PYTHONPATH=$PYTHONPATH:/home/svu/e0200920/volta_pypkg/lib/python3.8/site-packages

python3 train_classifier.py --seed 0 --pretrained_model --augment_data --weight_decay=$weight_decay --batch_size=$batch_size --init_lr=$init_lr --num_epochs 100 --output_dir=/hpctmp/e0200920/COND-CORAL-$weight_decay-$batch_size-$init_lr-$method_scale --method 5 --method_scale $method_scale --reweight_classes --data_dir /home/svu/e0200920/waterbird_complete95_forest2water2

EOF
