#!/bin/bash


export LANG=LANG="zh_CN.utf-8"
export LANGUAGE="zh_CN:zh:en_US:en"
export LC_ALL="zh_CN.utf-8" 

#cmgat:run_biokg72k_14_no_dgi.py
conda activate GNN
module load bazel/3.7.2
module load compilers/cuda/11.3
module load compilers/gcc/9.3.0
module load nccl/2.17.1-1_cuda11.3
cd /home/bingxing2/home/scx6266/GNN/new_CMGAT
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx6266/.conda/envs/geodiff/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0


CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 run_biokg72k_14_no_dgi.py --batch_size 2048 --dataset BioKG72k_14 --loss_function contrastive_loss --use_test 1 --test_name ChangeLR-UnitVec-BN-CLoss-part-sample-Tem1.0  --score_func conve --subgraph_loss_rate 0 --sample_mod part --temperature 1.0 --stop_num 100 --change_lr 0 --optimizer adam &




#cmgat:run_drkg17k-21_no_dgi.py #tmux a -t drkg3
ssh scx6266@paraai-n32-h-01-agent-15
conda activate GNN
module load bazel/3.7.2
module load compilers/cuda/11.3
module load compilers/gcc/9.3.0
module load nccl/2.17.1-1_cuda11.3
cd /home/bingxing2/home/scx6266/GNN/new_CMGAT
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx6266/.conda/envs/geodiff/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0


CUDA_VISIBLE_DEVICES=1 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 run_drkg17k-21_no_dgi.py --batch_size 1024 --dataset DRKG17k-21 --loss_function contrastive_loss --use_test 1 --test_name ChangeLR-UnitVec-BN-CLoss-part-sample-Tem1.0  --score_func conve --subgraph_loss_rate 0 --sample_mod part --temperature 1.0 --stop_num 100 --change_lr 0 --optimizer adam 


#SE-GNN£ºBioKG72k_14 tmux a -t se4
conda activate geodiff
module load bazel/3.7.2
module load compilers/cuda/11.3
module load compilers/gcc/9.3.0
module load nccl/2.17.1-1_cuda11.3
cd  /home/bingxing2/home/scx6266/GNN/SE-GNN
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx6266/.conda/envs/geodiff/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

CUDA_VISIBLE_DEVICES=2 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 /home/bingxing2/home/scx6266/GNN/SE-GNN/code/run.py dataset=BioKG72k_14 



#compgcn£ºBioKG72k_14 tmux a -t biokg

conda activate geodiff
module load bazel/3.7.2
module load compilers/cuda/11.3
module load compilers/gcc/9.3.0
module load nccl/2.17.1-1_cuda11.3
cd /home/bingxing2/home/scx6266/GNN/CompGCN
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx6266/.conda/envs/geodiff/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0


CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 mgpu_run_compgcn_biokg72k_14_no_dgi.py --dataset BioKG72k_14 