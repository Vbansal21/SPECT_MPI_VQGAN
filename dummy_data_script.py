import torch as t0
import torch as t1
import torch

shape = (1, 80, 80, 80)
gendat = lambda t: t.randint(0,255,shape)

for i in range(100):
    t0.manual_seed(0 + 2*i)
    t1.manual_seed(1 + 2*i)
    torch.save(gendat(t0),f"./dataset/train/normal/data{i}")
    torch.save(gendat(t1),f"./dataset/train/abnormal/data{i}")
    t0.manual_seed(0 + 2*i + 10000)
    t1.manual_seed(1 + 2*i + 10000)
    torch.save(gendat(t0),f"./dataset/test/normal/data{i}")
    torch.save(gendat(t1),f"./dataset/test/abnormal/data{i}")