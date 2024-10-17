
import os
import time
import torch
import numpy as np
import pandas as pd
from chronos import ChronosPipeline
from sklearn.cluster import KMeans

def chronos(ts, labels, nclust):
    pipeline = ChronosPipeline.from_pretrained(
            "./chronos-t5-small",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

    rep = []
    for t in ts:
        context = torch.tensor(t)
        embeddings, tokenizer_state = pipeline.embed(context)
        representation = torch.mean(embeddings, dim=1).to(torch.float32)
        rep.append(representation.squeeze(dim=0).cpu().detach().numpy())
        
    rep = np.array(rep)

    kmeans = KMeans(n_clusters=nclust, init='random', n_init=1).fit(rep)
    pred = kmeans.labels_

    return pred
  
