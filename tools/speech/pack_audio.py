import pickle
from tqdm import tqdm
import torch

with open("train_de_stt,train_nl_stt,train_es_stt,train_fr_stt,train_it_stt,train_pt_stt,train_ro_stt,train_ru_stt", "rb") as f:
    d = pickle.load(f)

sources = []
for i in tqdm(range(len(d))):
    item = torch.load("{}/item.{}".format("/mnt/hdd/qwang/must_data", i))



