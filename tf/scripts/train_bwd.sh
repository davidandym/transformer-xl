#$ -cwd
#$ -q gpu.q@@1080 -l gpu=4
#$ -l h_rt=120:00:00
#$ -N trans-xl-backward-eng
#$ -m bea
#$ -j y

module load cuda10.0/toolkit
module load cudnn/7.5.0_cuda10.0
module load nccl/2.4.2_cuda10.0

bash /exp/dmueller/transformer-xl/tf/scripts/enwik8_base_gpu_backward.sh eval
