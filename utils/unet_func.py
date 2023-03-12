import math
import torch

def timestep_embedding(timesteps, embedding_dim):

    timesteps = timesteps.float().unsqueeze(1)

    half_dim = embedding_dim // 2
    k = math.log(10000) / (half_dim - 1)
    channel = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -k).unsqueeze(0)
    channel = channel.to(timesteps.device)
    emb_ori = channel*timesteps
    emb = torch.cat([torch.sin(emb_ori), torch.cos(emb_ori)], dim=1)

    return emb



