from typing import Dict, Set, Tuple, Union

import numpy as np

import neighbourhood

CLUSTER_ID = int
VECTOR_ID = int
CLUSTERS = Dict[VECTOR_ID, CLUSTER_ID]
KNB = Dict[VECTOR_ID, Set]
NDF = Dict[VECTOR_ID, float]
R_KNB = Dict[VECTOR_ID, Set]


def nbc(vectors: np.array, k: int, reference_point: Union[None, np.array] = None) -> CLUSTERS:
    """A Neighborhood-Based Clustering Algorithm.

    :param vectors: numpy array in NxD dim (N - number of examples, D - dimensionality of each example)
    :param k: parameter of minimum required neighbours
    :param reference_point: (optional) reference point to use the Triangle Inequality optimization
    :return: Dictionary with indices as keys (based on the position in vectors param) and cluster index as value.
             Cluster index equal to -1 means a noise example.
    """
    clusters = {}
    for idx, _ in enumerate(vectors):
        clusters[idx] = -1
    noise = set()

    if reference_point is not None:
        knb, r_knb = neighbourhood.ti_k_neighbourhood(vectors, k, reference_point)
    else:
        knb, r_knb = neighbourhood.k_neighbourhood(vectors, k)

    ndf = neighbourhood.ndf(knb, r_knb)

    current_cluster_id = 0
    for idx, v in enumerate(vectors):
        if _has_cluster(idx, clusters) or not _is_dense_point(idx, ndf):
            continue
        clusters[idx] = current_cluster_id
        dense_points = set()

        for n_idx in knb[idx]:
            clusters[n_idx] = current_cluster_id
            if _is_dense_point(n_idx, ndf):
                dense_points.add(n_idx)

        while dense_points:
            dp = dense_points.pop()
            for n_idx in knb[dp]:
                if _has_cluster(n_idx, clusters):
                    continue
                clusters[n_idx] = current_cluster_id
                if _is_dense_point(n_idx, ndf):
                    dense_points.add(n_idx)

        current_cluster_id += 1

    for idx, v in enumerate(vectors):
        if clusters[idx] == -1:
            noise.add(idx)

    return clusters


def _is_dense_point(idx: VECTOR_ID, ndf: NDF) -> bool:
    return ndf[idx] >= 1


def _has_cluster(idx: VECTOR_ID, clusters: CLUSTERS) -> bool:
    return clusters[idx] != -1
