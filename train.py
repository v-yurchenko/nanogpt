import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

from gpt_model import *

learning_rate = 1e-3
max_iters = 20_000
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("start training, total steps = ", max_iters)
# training loop
t = tqdm(range(max_iters))
for iter in t:
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        t.set_postfix(train_loss=float(losses['train']), val_loss=float(losses['val']))
        # print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

if not os.path.exists('model'):
    os.system("mkdir model/")
    
print("saving trained model to: ", model_pt_file)
torch.save(model.state_dict(), model_pt_file)

print("Sample generation = ", inference('завещание', 2000))