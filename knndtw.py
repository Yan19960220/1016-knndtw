# -*- coding: utf-8 -*-
# @Time    : 3/21/21 2:21 PM
# @Author  : Yan
# @Site    : 
# @File    : knndtw.py
# @Software: PyCharm

import os
from collections import Counter
from itertools import groupby
from operator import itemgetter
from sklearn.metrics.pairwise import euclidean_distances
from data import getData, load_history_matrix, load_matrix_from_file
from tools import list2file
from dtw import FastDtw

OUTPUT_POS = True
POS_FILE = './pos.txt'
MATRIX_FILE = './dtw_matrix.bin'


class KnnDtw(object):

    def __init__(self, k, time_data, time, poses, dist="euclidean"):
        self.dist = dist
        self.distance_matrix = []
        self.poses = poses
        self.data = time_data
        self.time = time
        self.vote = []
        self.k = k

    def result(self):
        self.distance_matrix = self.cal_distance_matrix()
        self.vote = self.get_n_neighbors(self.distance_matrix)
        return self.votes(self.vote)

    def votes(self, vote_list):
        K_list, accuracy_list = self.calculate_a_k(self.k, vote_list)
        # K = 3
        L = self.invert2labelWithK(3, vote_list)
        m1 = []
        for i in range(len(L)):
            label = self.data[vote_list[i][0][0]][2]
            occ = L[i].count(label)
            c = 1 if occ > 1 else 0
            m1.append(c)
        accuracy_list.insert(0, sum(m1) / len(vote_list))
        K_list.insert(0, 3)
        # K = 1
        correct = 0
        for v in range(len(vote_list)):
            t = 0
            label = self.data[vote_list[v][0][0]][2]
            neighbor_label = self.data[vote_list[v][0][1]][2]
            c = 1 if neighbor_label == label else 0
            correct += c
        accuracy_list.insert(0, correct / len(vote_list))
        K_list.insert(0, 1)
        return K_list, accuracy_list

    def get_n_neighbors(self, distance_matrix):
        euc = []
        votes = []
        for t in self.data:
            t_id, t_label = t[0], t[2]

            for t1 in self.data:
                t1_id = t1[0]

                if t1_id != t_id:
                    distance = distance_matrix[t_id][t1_id]
                    if distance != 0:
                        euc.append((t_id, t1_id, distance))
        eucseg = [(list(group)) for key, group in groupby(euc, itemgetter(0))]
        for i in range(len(eucseg)):
            eucseg[i] = sorted(eucseg[i], key=itemgetter(2))
            votes.append(eucseg[i][:self.k])

        return votes

    def cal_distance_matrix(self):
        distance_matrix = []
        if self.dist == 'dtw':
            if load_history_matrix and \
                    os.path.exists(MATRIX_FILE):
                distance_matrix = load_matrix_from_file(MATRIX_FILE, len(self.poses))
            else:
                # Compute DTW distance matrix
                dtw_tools = FastDtw(self.time, self.time, None)
                distance_matrix = dtw_tools.dist_matrix()
                distance_matrix.tofile(MATRIX_FILE)
                if OUTPUT_POS:
                    list2file(POS_FILE, self.poses)
        elif self.dist == 'euclidean':
            # euclidean distance matrix
            distance_matrix = self.cal_euclidean(self.time)
        return distance_matrix

    def cal_euclidean(self, times_series):
        # return squareform(pdist(time, 'euclidean'))
        return euclidean_distances(times_series)

    def calculate_a_k(self, K, votes):
        k = []
        a = []
        for i in range(4, K + 1):
            L = self.invert2labelWithK(i, votes)
            # print(score(L, votes))
            a.append(self.score(L, votes))
            k.append(i)
        return k, a

    def invert2labelWithK(self, k, votes):
        L = []
        for v in range(len(votes)):
            label_list = []
            for i in range(k):
                label_list.append(self.data[votes[v][i][1]][2])
            L.append(label_list)
        return L

    def score(self, L2, votes):
        m = []
        for i in range(len(L2)):
            c = 0
            label = self.data[votes[i][0][0]][2]
            occ = L2[i].count(label)
            occ1 = Counter(L2[i]).most_common(1)

            if occ > 1:
                if occ1.__contains__((label, occ)):
                    c += 1

            m.append(c)
        return sum(m) / len(votes)
