# load the knowledge base

import pickle as pkl
import os

I = pkl.load(open('pndm_key10_idxs.pkl', 'rb'))

val_step = int(os.environ['VAL_STEP'])
guidance = float(os.environ['GUIDE'])

metainfo = pkl.load(open('metainfo.pkl', 'rb'))
trajs = pkl.load(open('trajs_kb.pkl', 'rb'))
    
print(len(metainfo))
print(len(trajs))

# load the sd model

from diffusers import ReSDPipeline
import torch

pipe = ReSDPipeline.from_pretrained('./stable-diffusion-v1-5', torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to("cuda")

# load metainfo of validation set

# stack the first few steps in the trajectory as key

import numpy as np
from tqdm import tqdm


key_step = 10

keys = [np.concatenate([t.flatten() for t in traj[key_step]]) for traj in tqdm(trajs)]

keys = np.stack(keys)

# stack the query trajectories

vmetas = []
queries = []

for split in range(8):
    vmeta, vtrajs = pkl.load(open(f'traj_val{split}.pkl', 'rb'))
    vmetas += vmeta
    for vtraj in tqdm(vtrajs):
        queries.append(vtraj[key_step].flatten())

queries = np.stack(queries)

pipe.set_progress_bar_config(disable=True)

cnt = 0

topk = 1

errs = []

BSIZE = 20
prompt_buf = []
vals_buf = []
id_buf = []
coco_prompt_buf = []

for i in tqdm(range(len(vmetas))):
    # print(I[i])
    # print(vmetas[i])
    # print(metainfo[I[i][0]])
    # print(metainfo[I[i][1]])
    
    topk_nbrs = []
    for j in range(topk):
        topk_nbrs.append(keys[I[i][j]])
    topk_nbrs = np.stack(topk_nbrs)
    
    def least_square(X, y):
        # minimize || X.dot(w) - y || ^2
        # w = (X.T.dot(X))^{-1}.dot(A.T).dot(y)
        return np.linalg.pinv(X).dot(y)
    
    w = least_square(topk_nbrs.T.astype('float32'), queries[i].astype('float32'))
    err = np.linalg.norm(topk_nbrs.T.dot(w) - queries[i])
    errs.append(err)
    #print(np.linalg.norm(topk_nbrs[0] - queries[i]))
    # print(w)
    # print(w.shape)
    # print(sum(w))
    vals = []
    for j in range(topk):
        vals.append(w[j] * trajs[I[i][j]][val_step - 1])
    val = torch.tensor(sum(vals), dtype=torch.float16, device=0)
    vals_buf.append(val)

    prompt = vmetas[i]['caption'] # make sure that it's the query caption
    prompt_buf.append(prompt)
    #coco_prompt_buf.append(metainfo[I[i][0]]['caption'])
    id_buf.append(vmetas[i]['image_id'])
    
    #print(f"prompt = {prompt}, retrieved = {metainfo[I[i][0]]['caption']}, coco_id = {metainfo[I[i][0]]['image_id']}, image_name = {vmetas[i]['image_id']}")
    
    if len(vals_buf) == BSIZE:
        generator = torch.Generator("cuda").manual_seed(1024)
        #print(prompt_buf)
        val = torch.cat(vals_buf)
        #print(val.shape)
        
        imgs = pipe(prompt_buf, head_start_latents=val, head_start_step=val_step, guidance_scale=guidance, generator=generator).images
        for (img, id) in zip(imgs, id_buf):
            img.save(f"reb/key{key_step}_val{val_step}_{guidance}/{id}.png")
        
        # imgs = pipe(coco_prompt_buf, guidance_scale=7.5, generator=generator).images
        # for (img, id) in zip(imgs, id_buf):
        #     img.save(f"tmp_imgs/{id[:-4]}_cooc.png")
        

        # coco_prompt_buf = []
        vals_buf = []
        prompt_buf = []
        id_buf = []