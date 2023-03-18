import os
import glob
import torch 
import numpy as np
import pickle

# доделать правильное скачивание
if not os.path.exists('data'):
    os.system("mkdir data/")
if not os.path.exists('data/data.tar.gz'):
    os.system("wget --no-check-certificate -O data/data.tar.gz -r https://github.com/v-yurchenko/nanogpt/blob/main/data/data.tar.gz?raw=true") #https://ipfs.sweb.ru/ipfs/QmSwzBsup91QiJZQS1xuSEq6NunJukvXLGTms17zgAm4Th/data.tar.gz")
if os.path.exists('data/data.tar.gz'):
    os.system("tar -xvf data/data.tar.gz --directory=data/ > /dev/null")
    os.system("rm data/data.tar.gz") 
    
# corpus_type = 'books'
# corpus_enc = 'utf8'

corpus_type = 'news'
corpus_enc = 'utf-8'

corpus_files  = f'data/{corpus_type}/*.txt'
train_pt_file = f'data/{corpus_type}_train.pt'
val_pt_file   = f'data/{corpus_type}_val.pt'
meta_pt_file  = f'data/{corpus_type}_meta.pkl'

text_files = glob.glob(corpus_files)
print('Всего файлов = ', len(text_files))

alphabet = set()
text_len = 0
text = ''
for file in text_files:
    with open(file, 'r', encoding=corpus_enc) as f:
        print(file)
        s = ' '.join(f.readlines())
        text += s + ' ' # ' '*64 # не будем захватывать текст из соседних статей? 
        text_len += len(s)
        alphabet.union(set(s))

print(f'Всего текстов: {len(text_files)}')
print(f'Длина текстов: {text_len % 2**20} Мб')

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Размер словаря до чистки = ', vocab_size)

allowed_symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё0123456789 .,-=" 
def clean_text(text):
    ret = ''
    for w in text:
        if w in allowed_symbols:
            ret+=w
    return ret

# так делаем, так как знаем что корпус маленький
text = clean_text(text)

chars = sorted(list(set(text).union(allowed_symbols)))
vocab_size = len(chars)
print('Размер словаря после чистки = ', vocab_size)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


data = torch.tensor(encode(text), dtype=torch.long)
n = len(data)
train_data = data[:int(n*0.9)]
val_data   = data[int(n*0.9):]



# encode both to integers
print(f"train has {len(train_data):,} tokens")
print(f"val has {len(val_data):,} tokens")

# export to bin files
print(f"saving {train_pt_file}")
torch.save(train_data, train_pt_file)
print(f"saving {val_pt_file}")
torch.save(val_data,   val_pt_file)

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
print(f"saving {meta_pt_file}")
with open(meta_pt_file, 'wb') as f:
    pickle.dump(meta, f)
    
print(f"saving parems")
parems_list = []
with open('data/parems_cleaned.txt', 'w') as fw:
    with open('data/parems.txt') as f:
        for line in f:
            l = clean_text(line.strip('\r\n'))
            if len(l) < 61:
                fw.write(l + '\n')

print(f"preparation is done")