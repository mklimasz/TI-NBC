from typing import Set, Dict, Tuple

import numpy as np

VECTOR_ID = int
KNB = Dict[VECTOR_ID, Set]
NDF = Dict[VECTOR_ID, float]
R_KNB = Dict[VECTOR_ID, Set]


def k_neighbourhood(vectors: np.array, k: int) -> Tuple[KNB, R_KNB]:
    knb = {}
    r_knb = {}
    for i in range(vectors.shape[0]):
        r_knb[i] = set()

    for idx1, v1 in enumerate(vectors):
        distances = []
        for idx2, v2 in enumerate(vectors):
            if idx1 != idx2:
                # TODO support other distances
                dist = np.linalg.norm(v2 - v1)
                distances.append((idx2, dist))
        distances.sort(key=lambda t: t[1])
        eps = distances[:k][-1][1]
        neighbours = {idx for (idx, dist) in distances if eps >= dist}
        knb[idx1] = neighbours
        for n in neighbours:
            r_knb[n].add(idx1)

    return knb, r_knb


def ndf(knb: KNB, r_knb: R_KNB) -> NDF:
    ndfs = {}
    for k in knb.keys():
        ndfs[k] = len(r_knb[k]) / len(knb[k])
    return ndfs
