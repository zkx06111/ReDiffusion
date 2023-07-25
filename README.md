# ReDiffusion

The code for the ICML 2023 paper [ReDi: Efficient Learning-Free Diffusion Inference via Trajectory Retrieval](https://arxiv.org/abs/2302.02285).

## Installation

1. Install diffusers from source.
```
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
```

2. Put the pipeline_re_sd.py in `./src/diffusers/pipelines` and import it in `./src/diffusers/__init__.py`.

## Knowledge Base Creation

1. Follow this (instruction)[https://github.com/facebookresearch/faiss/blob/main/INSTALL.md] to install `faiss-cpu`.

2. Run `traj.py` to construct the knowledge base.

## Image generation

1. `retrieve_val10.py` gives an example of generating images with ReDi one by one.

2. `retrieve_redi.py` gives an example of generating images with ReDi in batches.