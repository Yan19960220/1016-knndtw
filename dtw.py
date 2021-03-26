# -*- coding: utf-8 -*-
# @Time    : 3/21/21 2:43 PM
# @Author  : Yan
# @Site    : 
# @File    : dtw.py
# @Software: PyCharm

import sys
import numbers
import numpy as np
from fastdtw.fastdtw import __difference, __norm
from scipy.spatial.distance import squareform
from torch.autograd import Function
from collections import defaultdict
from scipy.spatial.distance import euclidean
from dtaidistance import dtw
from processBar import ProgressBar


# class DTW(object):
#     """ Based on the mark
#     """
#
#     def __init__(self, max_warping_window=pow(100, 2), subsample_step=1):
#         self.max_warping_window = max_warping_window
#         self.subsample_step = subsample_step
#
#     def dtw_distance(self, ts_a, ts_b, d=lambda x, y: abs(x - y)):
#         ts_a, ts_b = np.array(ts_a), np.array(ts_b)
#         M, N = len(ts_a), len(ts_b)
#         cost = sys.maxsize * np.ones(M, N)
#
#         cost[0, 0] = d[ts_a[0], ts_b[0]]
#         for i in range(1, M):
#             cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])
#
#         for j in range(1, N):
#             cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])
#
#         for i in range(1, M):
#             for j in range(max(1, i - self.max_warping_window),
#                            min(N, i + self.max_warping_window)):
#                 choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
#                 cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])
#
#         return cost[-1, -1]
#
#     def dist_matrix(self, x, y):
#         x, y = np.array(x), np.array(x)
#
#         dm_count = 0
#         if np.array_equal(x, y):
#             x_s = x.shape
#             dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)
#
#             for i in range(0, x_s[0] - 1):
#                 for j in range(i + 1, x_s[0]):
#                     dm[dm_count] = self.dtw_distance(x[i][::self.subsample_step],
#                                                      y[j][::self.subsample_step])
#
#             dm = squareform(dm)
#             return dm
#         else:
#             x_s, y_s = x.shape, y.shape
#             dm = np.zeros((x_s[0], y_s[0]))
#             # dm_size = x_s[0] * y_s[0]
#             for i in range(0, x_s[0]):
#                 for j in range(0, y_s[0]):
#                     dm[i, j] = self.dtw_distance(x[i, ::self.subsample_step],
#                                                  y[j, ::self.subsample_step])
#
#             return dm


class FastDtw(object):
    def __init__(self, x, y, dist, subsample_step=1):
        self.dist_2d = lambda a, b: sum((a - b) ** 2) ** 0.5
        self.subsample_step = subsample_step
        self.x = x
        self.y = y

    @staticmethod
    def dtw_distance(x, y):
        return dtw.distance_fast(x, y, use_pruning=True)
        # distance, _ = fastdtw_p(x, y, dist=euclidean)
        # return distance

    def dist_matrix(self):
        x, y = np.array(self.x), np.array(self.y)
        dm_count = 0
        if np.allclose(x, y):
            x_s = x.shape
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)
            p = ProgressBar(np.shape(dm)[0])
            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self.dtw_distance(x[i], y[j])
                    dm_count += 1
                    p.animate(dm_count)
            dm = squareform(dm)
            return dm
        else:
            x_s, y_s = np.shape(x), np.shape(y)
            dm = np.zeros((x_s[0], y_s[0]))
            dm_size = x_s[0] * y_s[0]
            p = ProgressBar(dm_size)
            for i in range(0, x_s[0] - 1):
                for j in range(i, y_s[0]):
                    # dm[i, j] = self.dtw_distance(x[i], y[j])
                    dm[dm_count] = self.dtw_distance(x[i], y[j])
                    dm_count += 1
                    p.animate(dm_count)
            dm = squareform(dm)
            return dm