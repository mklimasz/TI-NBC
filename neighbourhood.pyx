from typing import Set, Dict, Tuple, List

import numpy as pnp
cimport numpy as np
cimport cython
from sortedcontainers import SortedSet

VECTOR_ID = int
KNB = Dict[VECTOR_ID, Set]
NDF = Dict[VECTOR_ID, float]
R_KNB = Dict[VECTOR_ID, Set]

cdef extern from "math.h":
    double sqrt(double v)
    double pow(double base, double exponent)

@cython.boundscheck(False)
def distance(np.ndarray[np.float64_t, ndim=1] v1, np.ndarray[np.float64_t, ndim=1] v2) -> float:
    cdef Py_ssize_t i
    cdef Py_ssize_t n = v1.shape[0]
    cdef double dist = 0.0
    for i in range(n):
        dist += pow(v1[i] - v2[i], 2.0)
    return sqrt(dist)


class _Point:

    def __init__(self, idx, vector, dist, preceding=None, following=None):
        self.idx = idx
        self.vector = vector
        self.dist = dist
        self.preceding = preceding
        self.following = following

    def __repr__(self):
        return "Point(idx: {}, vector: {}, dist: {}".format(self.idx, self.vector, self.dist)

    def __str__(self):
        return "idx: {}, vector: {}, dist: {}".format(self.idx, self.vector, self.dist)


def k_neighbourhood(vectors: np.ndarray, k: int) -> Tuple[KNB, R_KNB]:
    knb, r_knb = _init(vectors)

    for idx1, v1 in enumerate(vectors):
        neighbour_candidates = []
        for idx2, v2 in enumerate(vectors):
            if idx1 != idx2:
                dist = distance(v1, v2)
                neighbour_candidates.append((idx2, dist))
        neighbour_candidates.sort(key=lambda t: t[1])
        eps = neighbour_candidates[:k][-1][1]

        neighbours = set()
        for (i, d) in neighbour_candidates:
            if d > eps:
                break
            neighbours.add(i)
        _fill(knb, r_knb, idx1, neighbours)
    return knb, r_knb

def ti_k_neighbourhood(vectors: np.ndarray, k: int, reference_point: np.ndarray) -> Tuple[KNB, R_KNB]:
    knb, r_knb = _init(vectors)

    points = _ti(vectors, reference_point)
    for point in points:
        idx = point.idx
        neighbours = _ti_neighbours(point, k)
        _fill(knb, r_knb, idx, neighbours)

    return knb, r_knb

def ndf(knb: KNB, r_knb: R_KNB) -> NDF:
    ndfs = {}
    for k in knb.keys():
        ndfs[k] = len(r_knb[k]) / len(knb[k])
    return ndfs

def _init(vectors):
    knb = {}
    r_knb = {}
    for i in range(vectors.shape[0]):
        r_knb[i] = set()
    return knb, r_knb

def _fill(knb, r_knb, vector_idx, neighbours):
    knb[vector_idx] = neighbours
    for n in neighbours:
        r_knb[n].add(vector_idx)

def _ti_neighbours(point: _Point, k):
    bp = point
    fp = point
    backward_search = bp.preceding
    bp = bp.preceding
    forward_search = fp.following
    fp = fp.following

    neighbour_candidates = SortedSet(key=lambda x: x[1])

    bp, fp, backward_search, forward_search = _candidate_nbs(backward_search,
                                                             forward_search,
                                                             neighbour_candidates,
                                                             k,
                                                             point,
                                                             bp,
                                                             fp)
    eps = neighbour_candidates[-1][1]
    # TODO make generic verify function
    eps = _verify_backward(point, bp, backward_search, neighbour_candidates, k, eps)
    eps = _verify_forward(point, fp, forward_search, neighbour_candidates, k, eps)
    return [n[0].idx for n in neighbour_candidates]

def _verify_forward(p: _Point, fp: _Point, forward_search: bool, neighbour_candidates: SortedSet, k: int, eps: float):
    while forward_search and (p.dist - fp.dist) <= eps:
        dist = distance(fp.vector, p.vector)
        if dist < eps:
            i = len([e for e in neighbour_candidates if e[1] == eps])
            if len(neighbour_candidates) - i >= k - 1:
                for e in neighbour_candidates:
                    if e[1] == eps:
                        neighbour_candidates.remove(e)
                neighbour_candidates.add((fp, dist))
                eps = neighbour_candidates[-1][1]
            else:
                neighbour_candidates.add((fp, dist))
        elif dist == eps:
            neighbour_candidates.add((fp, dist))
        forward_search = fp.following
        fp = fp.following
    return eps

def _verify_backward(p: _Point, bp: _Point, backward_search: bool, neighbour_candidates: SortedSet, k: int, eps: float):
    while backward_search and (p.dist - bp.dist) <= eps:
        dist = distance(bp.vector, p.vector)
        if dist < eps:
            i = len([e for e in neighbour_candidates if e[1] == eps])
            if len(neighbour_candidates) - i >= k - 1:
                for e in neighbour_candidates:
                    if e[1] == eps:
                        neighbour_candidates.remove(e)
                neighbour_candidates.add((bp, dist))
                eps = neighbour_candidates[-1][1]
            else:
                neighbour_candidates.add((bp, dist))
        elif dist == eps:
            neighbour_candidates.add((bp, dist))
        backward_search = bp.preceding
        bp = bp.preceding
    return eps

def _candidate_nbs(backward_search: bool,
                   forward_search: bool,
                   neighbour_candidates: SortedSet,
                   k: int,
                   p: _Point,
                   bp: _Point,
                   fp: _Point):
    i = 0
    while forward_search and backward_search and i < k:
        if p.dist - bp.dist < fp.dist - p.dist:
            dist = distance(bp.vector, p.vector)
            neighbour_candidates.add((bp, dist))
            backward_search = bp.preceding
            bp = bp.preceding
        else:
            dist = distance(fp.vector, p.vector)
            neighbour_candidates.add((fp, dist))
            forward_search = fp.following
            fp = fp.following
        i += 1
    while forward_search and i < k:
        dist = distance(fp.vector, p.vector)
        i += 1
        neighbour_candidates.add((fp, dist))
        forward_search = fp.following
        fp = fp.following
    while backward_search and i < k:
        dist = distance(bp.vector, p.vector)
        i += 1
        neighbour_candidates.add((bp, dist))
        backward_search = bp.preceding
        bp = bp.preceding
    return bp, fp, backward_search, forward_search

def _ti(vectors: np.ndarray,
        reference_point: np.ndarray) -> List[_Point]:
    rp_dist = []
    for idx, v in enumerate(vectors):
        dist = distance(v, reference_point)
        rp_dist.append(dist)
    arg_sorted_rp_dist = pnp.argsort(rp_dist)
    points = []
    for i, vector_id in enumerate(arg_sorted_rp_dist):
        if i == 0:
            points.append(_Point(vector_id, vectors[vector_id], rp_dist[vector_id]))
        else:
            point = _Point(vector_id, vectors[vector_id], rp_dist[vector_id], preceding=points[i - 1])
            points.append(point)
            points[i - 1].following = point
    return points
