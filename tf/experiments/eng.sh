#! /usr/bin/env bash

# ADJUST THESE
TIME_LIMIT=24:00:00
REV=`git rev-parse --short HEAD`
TRAINER=`realpath scripts/enwik8_base_gpu_forward.sh`

set -e
set -u

if [ ! -f ${TRAINER} ]; then
    echo "Can't find trainer: ${TRAINER}"
	exit
fi

CHANGES=`git status --porcelain --untracked-files=no`
if [[ $CHANGES ]]; then
    echo "Commit changes first:"
    echo "$CHANGES"
    exit
fi

JOB_NAME="trans-xl-forward-eng"
JOB_SCRIPT="${JOB_NAME}.sh"

cat >${JOB_SCRIPT} <<EOL

#$ -cwd
#$ -q gpu.q -l gpu=1,h=!r8n*
#$ -l h_rt=${TIME_LIMIT}
#$ -N ${JOB_NAME}
#$ -m bea
#$ -j y

module load cuda10.0/toolkit
module load cudnn/7.5.0_cuda10.0
module load nccl/2.4.2_cuda10.0


python ${TRAINER} train

EOL

echo "Submitting to queue: ${JOB_SCRIPT}"
qsub -o ${CHECKS}/out -e ${CHECKS}/err ${JOB_SCRIPT}

#eof
