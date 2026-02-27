#!/bin/bash
#SBATCH --job-name=compute_info_oven
#SBATCH --output=/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/self-elicit/logs/%A/%a.out
#SBATCH --error=/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/self-elicit/logs/%A/%a.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --mem=20G
#SBATCH --partition=lrd_all_serial
#SBATCH --cpus-per-task=4
#SBATCH --account=AIFAC_S02_096
#SBATCH --time=00:10:00
# #SBATCH --qos=boost_qos_dbg


source /leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge/pyenvs/MLLM_challenge/bin/activate
cd /leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge/utils

input_path=$1

python amber_disc_eval/amber_test.py --input_path "$input_path"
