#!/bin/bash

#$ -cwd
#$ -q all.q 
#$ -l num_proc=1,h_rt=12:00:00,mem_free=32G
#$ -m bea
#$ -N bwd-rus-preproc
#$ -j y

# sh scripts/rus_base_gpu_forward.sh train_data
sh scripts/dewiki_base_gpu.sh train_data
