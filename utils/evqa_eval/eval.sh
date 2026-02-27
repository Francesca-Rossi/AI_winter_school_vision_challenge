#!/bin/bash
#SBATCH --job-name=evqa_eval
#SBATCH --partition=boost_usr_prod
# #SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_S02_096
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --time=00:10:00
#SBATCH --exclude=lrdn3443
#SBATCH --output=/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/grpo/logs/%A/%x.out
#SBATCH --error=/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/grpo/logs/%A/%x.err
mkdir -p /leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/grpo/logs/$SLURM_ARRAY_JOB_ID

module load cuda/11.8
module load profile/deeplrn


source /leonardo_scratch/fast/tra26_minwinsc/pyenvs/MLLM_challenge/bin/activate

export PYTHONPATH=.
export CUDA_HOME=/leonardo/prod/opt/compilers/cuda/11.8/none

# See https://www.ou.edu/oscer/support/python/python-with-tensorflow
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib


# Verifica che TensorFlow veda la GPU
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

cd /leonardo_scratch/fast/tra26_minwinsc/


input_path=$1
python evqa_compute_metrics.py \
    --input_path "$input_path"
