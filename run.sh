#!/bin/bash
#SBATCH --job-name=gpu_mem_test_py
#SBATCH --output=gpu_mem_test_py_%j.out
#SBATCH --error=gpu_mem_test_py_%j.err
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=11G
#SBATCH --account=g2tfg11
#SBATCH -p pg2tfg11
#SBATCH --qos=q_pg2tfg11

echo "========= GPU asignada por Slurm ========="
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<no-set>}"
nvidia-smi
echo "=========================================="

rm -r "/home/amartinez/TFG/dataset/images/train/"
rm -r "/home/amartinez/TFG/dataset/images/val/"
rm -r "/home/amartinez/TFG/dataset/labels/train/"
rm -r "/home/amartinez/TFG/dataset/labels/val/"

python main.py