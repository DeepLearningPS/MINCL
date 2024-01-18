<h1 align="center">
  NC-KGE
</h1>

<h4 align="center">Node-based Knowledge Graph Contrastive Learning for Medical Relationship Prediction</h4>



<h2 align="center">
  Overview of NC-KGE
  <img align="center"  src="https://github.com/DeepLearningPS/NC-KGE/blob/main/NC-KGE/image/cl.png" alt="...">
</h2>

<!--
# We add some experiments in the appendix of the paper. Details can be found in https://github.com/DeepLearningPS/NC-KGE/blob/main/NC-KGE.pdf
-->


### Dependencies

- Compatible with PyTorch 1.x , Python 3.8, tensorflow can be a cpu or gpu version
- Dependencies can be installed using `requirements.txt`.

### Dataset:

- We use  PharmKG8k-28, DRKG17k-21 and BioKG72k-14 datasets for knowledge graph link prediction. 
- These are included in the `../data` directory.

#### The detailed description of datasets is as follows:

- FB15k-237 [Observed versus latent features for knowledge base and text inference](https://aclanthology.org/W15-4007/) dataset contains knowledge base relation triples and textual mentions of Freebase node pairs, as used in the work published in \cite{toutanova2015observed}. The knowledge base triples are a subset of the FB15K \cite{bordes2013translating}, originally derived from Freebase. The inverse relations are removed in FB15k-237. To obtain node descriptions and types, we employ the datasets made available by \cite{daza2021inductive}. 

- {WN18RR} \cite{dettmers2018convolutional} is created from WN18 \cite{bordes2013translating}, which is a subset of WordNet. WN18 consists of 18 relations and 40,943 nodes. However, many text triples obtained by inverting triples from the training set. Thus WN18RR dataset \cite{dettmers2018convolutional} is created to ensure that the evaluation dataset does not have inverse relation test leakage. To obtain node descriptions and types, we employ the datasets made available by \cite{daza2021inductive}. 
	
- {PharmKG8k-28} is built based on PharmKG dataset \cite{10.1093/bib/bbaa344}. PharmKG is a multi-relational, attributed biomedical KG, composed of more than 500 000 individual interconnections between genes, drugs and diseases, with 29 relation types over a vocabulary of $\sim$ 8000 disambiguated nodes.  Each node in PharmKG is attached with heterogeneous, domain-specific information obtained from multi-omics data, i.e. gene expression, chemical structure and disease word embedding, while preserving the semantic and biomedical features. We obtained PharmKG8k-28 dataset after simple deduplication cleaning of PharmKG.


- {DRKG17k-21} is built based on Drug Repositioning Knowledge Graph (DRKG) \cite{drkg2020}. DRKG is a comprehensive biological knowledge graph relating genes, compounds, diseases, biological processes, side effects and symptoms.  DRKG includes information from six existing databases including DrugBank, Hetionet, GNBR, String, IntAct and DGIdb, and data collected from recent publications particularly related to Covid19. DRKG17k-21 is a subset that takes the top 10\% of the DRKG.
	
- {BioKG72k-14} is built based on BioKG dataset \cite{DBLP:conf/cikm/WalshMN20}. BioKG is a new more standardised and reproducible biological knowledge graph which provides a compilation of curated relational data from open biological databases in a unified format with common, interlinked identifiers. BioKG can be used to train and assess the relational learning models in various tasks related to pathway and drug discovery. We obtained BioKG72k-14 dataset after simple deduplication cleaning of BioKG.

- Fast convergence of node-based contrastive learning plays an important role in some applications. DRKG is a large biomedical dataset, and DRKG17k-21 is a subset of which we take the top 10\%. In order to further verify the efficiency of NC-KGE on DRKG, we randomly selected three subsets on it, namely DRKG35k-107, DRKG38k-107 and DRKG40k-107.



### Training model:

- Install all the requirements from `requirements.txt.`
- Commands for reproducing the reported results on link prediction:


```shell

# PharmKG8k-28
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 run_nckge_no_dgi.py  --dataset New_PharmKG8k-28 --loss_function contrastive_loss --score_func conve --temperature 1.0 --stop_num 50 --model_dir ./pharm8k_model_save  --log_dir ./pharm8k_log

# DRKG17k-21
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 run_nckge_no_dgi.py  --dataset DRKG17k-21 --loss_function contrastive_loss --score_func conve --temperature 1.0 --stop_num 50 --model_dir ./drkg17k-21_model_save  --log_dir ./drkg17k-21_log

# BioKG72k-14
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 run_nckge_no_dgi.py  --dataset BioKG72k-14 --loss_function contrastive_loss --score_func conve --temperature 1.0 --stop_num 50 --model_dir ./biokg_model_save  --log_dir ./biokg_log
  
```


### Citation:
