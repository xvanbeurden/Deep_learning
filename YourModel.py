import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url
from typing import Any
import torch.nn.functional as F


class NetVLAD(nn.Module):
    '''Only need NetVLAD class.
    CNN layers were imported'''

    def __init__(self, config, num_clusters=64, dim=128,
                 normalize_input=True):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def init_params(self, clsts, traindescs):
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(traindescs)
        del traindescs
        dsSq = np.square(knn.kneighbors(clsts, 2)[1])
        del knn
        self.alpha = (-np.log(0.01) / np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        del clsts, dsSq

        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    # based on lyakaap: https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C = x.shape[:2]

        x = F.normalize(x, p=2, dim=1)  # design choice: we always normalize before NETVLAD

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[C:C + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C + 1, :].unsqueeze(2)
            vlad[:, C:C + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        x = vlad

        return x