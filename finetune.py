import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

from gpt_model import *

print(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# Finetune: see Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning ratemax_iters = 10_000
# finetune at constant LR
max_iters = 15_000
learning_rate = 5e-4
decay_lr = False

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model.load_state_dict(torch.load(model_pt_file))

parems_list = []
with open('data/parems_cleaned.txt') as f:
    for line in f:
        parems_list.append(line.strip('\r\n'))
        
n = int(0.9*len(parems_list)) # first 90% will be train, rest val
indicies = [i for i in range(len(parems_list))]
np.random.shuffle(parems_list)
# parems_list = parems_list[]
parems_train_data = parems_list[:n]
parems_val_data   = parems_list[n:]

def get_parems_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = parems_train_data if split == 'train' else parems_val_data
    bs = 64
    # take random parems
    ix = torch.randint(0, len(data), (bs, ))
    x = []
    y = []
    for i in ix:
        # divide it randomly - may not from 5????
        j = torch.randint(5, len(data[i]), (1,))
        prompt = data[i][:j]+'ПП='+data[i][j:]
        prompt = prompt[:block_size]+''.join([' ']*(block_size-len(prompt))) + ' '
        prompt_x = prompt[:block_size]
        prompt_y = prompt[1:block_size+1]
        x.append(torch.tensor(encode(prompt_x), dtype=torch.long))
        y.append(torch.tensor(encode(prompt_y), dtype=torch.long))
                
    x = torch.stack(x)
    y = torch.stack(y)
    x, y = x.to(device), y.to(device)
    return x, y

def get_reverse_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (64, ))
    jx = torch.randint(1,  block_size//2, (64,))
    x = []
    y = []
    for i,j in zip(ix, jx):
        sent = decode(data[i:i+j].tolist())
        prompt = (sent+"ПФ="+sent[::-1])
        prompt = prompt[:block_size]+''.join([' ']*(block_size-len(prompt))) + ' '
        prompt_x = prompt[:block_size]
        prompt_y = prompt[1:block_size+1]
        x.append(torch.tensor(encode(prompt_x), dtype=torch.long))
        y.append(torch.tensor(encode(prompt_y), dtype=torch.long))
                
    x = torch.stack(x)
    y = torch.stack(y)
    x, y = x.to(device), y.to(device)
    # ix = torch.randint(len(data) - block_size, (batch_size,))
    x, y = x.to(device), y.to(device)
    return x, y

def get_batch(split):
    task0_x, task0_y = get_main_task_batch(split) # не забываем про основную задачу
    task1_x, task1_y = get_reverse_batch(split)   # учим на переворот
    task2_x, task2_y = get_parems_batch(split)    # учим на пословицы
    return torch.cat([
                task0_x, 
                task1_x, 
                task2_x], dim=0), \
           torch.cat([
                task0_y, 
                task1_y, 
                task2_y], dim=0) 

print("Finetuning language model")
print("Start training, total steps = ", max_iters)
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

print("Saving finetune model to: ",model_finetuned_pt_file)
torch.save(model.state_dict(), model_finetuned_pt_file)