<h1 align="center">
  NC-KGE
</h1>

<h4 align="center">Node-based Knowledge Graph Contrastive Learning for Medical Relationship Prediction</h4>



<h2 align="center">
  Overview of NC-KGE
  <img align="center"  src="./image/cl.png" alt="...">
</h2>

# We add some experiments in the appendix of the paper. Details can be found in https://github.com/DeepLearningPS/NC-KGE/blob/main/NC-KGE.pdf


### Dependencies

- Compatible with PyTorch 1.x , Python 3.8, tensorflow can be a cpu or gpu version
- Dependencies can be installed using `requirements.txt`.

### Dataset:

- We use  PharmKG8k-28, DRKG17k-21 and BioKG72k-14 datasets for knowledge graph link prediction. 
- These are included in the `../data` directory. 

### Training model:

- Install all the requirements from `requirements.txt.`
- Commands for reproducing the reported results on link prediction:


```shell

# PharmKG8k-28
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 run_nckge_no_dgi.py  --dataset PharmKG8k-28 --loss_function contrastive_loss --score_func conve --temperature 1.0 --stop_num 50 --model_dir ./pharm8k_model_save  --log_dir ./pharm8k_log

# DRKG17k-21
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 run_nckge_no_dgi.py  --dataset DRKG17k-21 --loss_function contrastive_loss --score_func conve --temperature 1.0 --stop_num 50 --model_dir ./drkg17k-21_model_save  --log_dir ./drkg17k-21_log

# BioKG72k-14
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 run_nckge_no_dgi.py  --dataset BioKG72k-14 --loss_function contrastive_loss --score_func conve --temperature 1.0 --stop_num 50 --model_dir ./biokg_model_save  --log_dir ./biokg_log
  
```


### Citation:
