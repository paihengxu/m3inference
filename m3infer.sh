#!/bin/bash
#$ -cwd
#$ -j y -o /export/c10/pxu/gender/m3inference/test_training.log
#$ -m bea
#$ -M paiheng@jhu.edu
#$ -l mem_free=15G,ram_free=15G,gpu=1,hostname=!b0[123456789]*&!b10*&!c20*
#$ -pe smp 1
#$ -V
#$ -q g.q
#$ -N "m3infer"
#export PATH="/export/c10/pxu/venv3.7/bin:$PATH"
source /export/c10/pxu/venv3.7/bin/activate
echo $PATH
export PYTHONPATH=${PYTHONPATH}:.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/NVIDIA/cuda-9.0/lib64/
python -V
CUDA_VISIBLE_DEVICES=`free-gpu -n 1` python train_text.py
