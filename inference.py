import os
import pickle
import numpy as np
import torch

from gpt_model import *
print('Loading: ', model_finetuned_pt_file)
model.load_state_dict(torch.load(model_finetuned_pt_file))

with open('test_cases.txt', 'r') as f: 
    for sent in f:
        output = inference(sent.strip('\r\n'))
        print("Prompt : ", sent)
        print("Predict: ", output)
        print("="*80)