import torch
import numpy as np
import pandas as pd

from momentfm import MOMENTPipeline
from sklearn.cluster import KMeans


def moment(ts, labels, nclust):
    model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={"task_name": "embedding"},
        )
    model.init()
    model.cuda()

    rep = []
    for t in ts:
        context = torch.tensor(t).unsqueeze(0).unsqueeze(0).to(torch.float32)
        embeddings = model(context.cuda())
        
        representation = embeddings.embeddings.squeeze(0).detach().cpu().numpy()
        rep.append(representation)
        
    rep = np.array(rep)

    kmeans = KMeans(n_clusters=nclust, init='random', n_init=1).fit(rep)
    pred = kmeans.labels_

    return pred
