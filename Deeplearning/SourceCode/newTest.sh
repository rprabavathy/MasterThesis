#!/bin/bash
#SBATCH --job-name=newtest
#SBATCH --partition=gpu
#SBATCH --account=zim_gpu

#SBATCH -N 1
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 16     # Always request 16 cores per GPU. This line can be omitted
#SBATCH --gres=gpu:4

#SBATCH --time=0-72:0:00
#SBATCH --output=./output3.out
#SBATCH --error=./error3.err

module load 2023a GCCcore/13.2.0 
module load NCCL/2.20.5-CUDA-12.4.0

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#watch -n 1 nvidia-smi

#export CUDA_VISIBLE_DEVICES=0,1,2

#Check CUDA version
echo "CUDA version:"
nvcc --version

# Check NVIDIA GPU status
echo "NVIDIA GPU status:"
#nvidia-smi
nvidia-smi 

#watch -n 1 'nvidia-smi --query-gpu=index,timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits > gpu_monitoring.log'

#nohup nvidia-smi --query-gpu=index,timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits -l 1 > gpu_monitoring.log 2>&1 &

# Check environment variables
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "XLA_FLAGS: $XLA_FLAGS"


source myenv/bin/activate
{
    echo "Training Model..."
    start_time=$(date +%s)
    HYDRA_FULL_ERROR=1 srun python3 -m src.scripts.Ntrain # --multirun 
    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    echo "Execution time: $execution_time seconds"
    
}| tee output3.log

