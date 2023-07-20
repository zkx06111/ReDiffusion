import os
split = int(os.environ['CUDA_VISIBLE_DEVICES'])
print(os.environ['CUDA_VISIBLE_DEVICES'])

from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained('./stable-diffusion-v1-5', torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to("cuda")

def gen_traj(prompt, pipe):
    generator = torch.Generator("cuda").manual_seed(1024)

    def collect_latents(step, timestep, latents, traj):
        traj.append(latents.cpu().numpy())

    traj = []
    images = pipe(prompt, callback_steps=1, callback=lambda step, timestep, latents: collect_latents(step, timestep, latents, traj), guidance_scale=7.5, generator=generator).images

    return traj

import json

train2014 = json.load(open('annotations/captions_train2014.json'))
len(train2014['images'])

cnt = {}
for tup in train2014['annotations']:
    iid = tup['image_id']
    if iid not in cnt:
        cnt[iid] = tup
        
import random
import tqdm

random.seed(19260817)
ids = sorted(list(cnt))
random.shuffle(ids)

STEP = 10350
print(f'split = {split}')
trajs = []
metainfo = []
pipe.set_progress_bar_config(disable=True)
print(f'interval = {STEP * split}, {STEP * (split + 1)}')
with torch.no_grad():
    for x in tqdm.tqdm(ids[STEP * split: STEP * (split + 1)]):
        trajs.append(gen_traj(cnt[x]['caption'], pipe))
        metainfo.append(cnt[x])

import numpy
import pickle
pickle.dump((metainfo, trajs), open(f'traj{split}.pkl', 'wb'))