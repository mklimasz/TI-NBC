from typing import Dict, Tuple, Set

import numpy as np

import nb

CLUSTER_ID = int
CLUSTERS = Dict[nb.VECTOR_ID, CLUSTER_ID]
NOISE = Set[nb.VECTOR_ID]


def nbc(vectors: np.array, k: int) -> Tuple[CLUSTERS, NOISE]:
    cluster_no = {}
    for idx, _ in enumerate(vectors):
        cluster_no[idx] = -1
    noise = set()

    knb, r_knb = nb.k_neighbourhood(vectors, k)
    ndf = nb.ndf(knb, r_knb)

    current_cluster_id = 0
    for idx, v in enumerate(vectors):
        if _has_cluster(idx, cluster_no) or not _is_dense_point(idx, ndf):
            continue
        cluster_no[idx] = current_cluster_id
        dense_points = set()

        for n_idx in knb[idx]:
            cluster_no[n_idx] = current_cluster_id
            if _is_dense_point(n_idx, ndf):
                dense_points.add(n_idx)

        while dense_points:
            p = dense_points.pop()
            for n_idx in knb[p]:
                if _has_cluster(idx, cluster_no):
                    continue
                cluster_no[n_idx] = current_cluster_id
                if _is_dense_point(n_idx, ndf):
                    dense_points.add(n_idx)

        current_cluster_id += 1

    for idx, v in enumerate(vectors):
        if cluster_no[idx] == -1:
            noise.add(idx)

    return cluster_no, noise


def _is_dense_point(idx: int, ndf: Dict[int, float]) -> bool:
    return ndf[idx] >= 1


def _has_cluster(idx: int, clusters: Dict[int, int]) -> bool:
    return clusters[idx] != -1
