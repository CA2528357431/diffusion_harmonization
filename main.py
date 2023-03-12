import torch
import torch.nn as nn
import math
import numpy

tar = torch.randn(6,3,4,4)
res = torch.randn(6,3,4,4)

t = nn.AdaptiveAvgPool2d((1,1))

tar = t(tar).reshape(tar.shape[:2])
res = t(res).reshape(tar.shape[:2])

m = nn.CosineSimilarity()

x = 1-m(tar,res)

print(x)



x1 = torch.randn(6,3,4,4)
x2 = torch.randn(6,3,8,8)

print(t(x1).shape)
print(t(x2).shape)