#!/bin/bash

#设置显卡数量
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scx6266

#加载环境 #切换项目目录


#cmgat:run_biokg72k_14_no_dgi.py
conda activate GNN
module load bazel/3.7.2
module load compilers/cuda/11.3
module load compilers/gcc/9.3.0
module load nccl/2.17.1-1_cuda11.3
cd /home/bingxing2/home/scx6266/GNN/new_CMGAT
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx6266/.conda/envs/geodiff/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0


CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 2 --rdzv_id 2 run_biokg72k_14_no_dgi.py --batch_size 2048 --dataset BioKG72k_14 --loss_function contrastive_loss --use_test 1 --test_name ChangeLR-UnitVec-BN-CLoss-part-sample-Tem1.0  --score_func conve --subgraph_loss_rate 0 --sample_mod part --temperature 1.0 --stop_num 1000 --change_lr 0 --optimizer adam --save_path LP_ChangeLR-UnitVec-BN-CLoss-part-sample-Tem1.0_notext_conve_contrastive_loss_BioKG72k_14_CLcmgat_08_07_2023_05_01_27


#sbatch --gpus=2 -p vip_gpu_scx6266 -w paraai-n32-h-01-agent-9  cmgat_run1.sh


