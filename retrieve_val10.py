from tqdm import tqdm
import pickle as pkl

I = pkl.load(open('key_query_indices.pkl', 'rb'))

from diffusers import ReSDPipeline
import torch

import os
split = int(os.environ['CUDA_VISIBLE_DEVICES'])
print(os.environ['CUDA_VISIBLE_DEVICES'])

STEP = 573

pipe = ReSDPipeline.from_pretrained('./stable-diffusion-v1-5', torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to('cuda')

pipe.set_progress_bar_config(disable=True)

trajs = pkl.load(open('trajs_kb.pkl', 'rb'))

vmetas = []
for split in range(8):
    vmeta, _ = pkl.load(open(f'traj_val{split}.pkl', 'rb'))
    vmetas += vmeta

print("VMETAS loaded")

key_step = 10

for i in tqdm(range(I.shape[0])):
    keyidx = I[i][0]
    traj = torch.tensor(trajs[keyidx], dtype=torch.float16).cuda()
    #print(vmetas[i])
    prompt = vmetas[i]['caption'] # make sure that it's the query caption
    generator = torch.Generator("cuda").manual_seed(1024)
    #print(prompt)
    img = pipe(prompt, head_start_latents=traj[key_step-1], head_start_step=10, guidance_scale=7.5, generator=generator).images[0]
    img.save(f"traj_key_traj10_val_ttt/{vmetas[i]['image_id']}.png")
    #break