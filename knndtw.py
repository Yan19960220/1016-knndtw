# -*- coding: utf-8 -*-
# @Time    : 3/21/21 2:21 PM
# @Author  : Yan
# @Site    : 
# @File    : knndtw.py
# @Software: PyCharm

import os
import numpy as np
from collections import Counter
from itertools import groupby
from operator import itemgetter
from sklearn.metrics.pairwise import euclidean_distances
from data import load_history_matrix, load_matrix_from_file, POS_FILE, OUTPUT_POS, list2array
from tools import list2file, most_common, most_common_row, most_frequent
from dtw import FastDtw

MATRIX_FILE = './dtw_matrix.bin'


class KnnDtw(object):

    def __init__(self, k, data, dist="euclidean",
                 random_range=[50]):
        self.k = k
        if isinstance(data, dict):
            self.random_range = random_range
            self.result = {}
            for i in random_range:
                self.result[i] = []
            self.dist = dist
            self.distance_matrix = []
            self.data = data
            self.label = []
            self.poses = []
            self.time = []
            self.vote = {}
            self.label = {}

    def merge_view(self):
        acc_range_dict = {}
        for ran in self.random_range:
            print("range - " + str(ran))
            acc_range_dict[ran] = self.merge_vote(ran)
        return acc_range_dict

    def get_vote_label_pos_dict(self):
        self.vote, self.label, self.poses = {}, {}, {}
        for i in self.random_range:
            self.vote[i] = self.get_n_neighbors_dict(i)[i]
            self.label[i] = self.get_label_dict(i)[i]
            self.poses[i] = self.get_pos_dict(i)[i]

    def index2label(self, index_list, range_value, index_value):
        label_list = []
        for item in index_list:
            # d = self.label[range_value][index_value][item[1]]
            label_list.append(self.label[range_value][index_value][item[1]])
        return label_list

    def merge_vote(self, range_value):
        self.get_vote_label_pos_dict()
        L_d = self.vote.get(range_value)
        correct_list, predict_result = [], []
        for i in range(1, self.k + 1):
            correct_label, predict_list = self.get_pos_predict_result_list(L_d, range_value, i)
            correct = self.cal_a(correct_label, predict_list)
            correct_list.append(correct / len(correct_label))
        return correct_list

    def get_pos_predict_result_list(self, L_d, range_value, k):
        predict_list = self.get_predict_label_list(L_d, k, range_value)
        correct_label = self.get_correct_label_list(range_value)
        return correct_label, predict_list

    def get_correct_label_list(self, range_value):
        correct_label = self.label[range_value][0]
        return correct_label

    def get_predict_label_list(self, L_d, k, range_value):
        predict_all_list = self.get_all_predict_label_list(L_d, k, range_value)
        predict_list = most_common_row(predict_all_list)
        return predict_list

    def get_all_predict_label_list(self, L_d, k, range_value):
        predict_all_list = []
        # data = L_d[range_value]
        for n in range(len(L_d)):
            predict_list = []
            for n_0 in range(len(L_d[n])):
                d = L_d[n][n_0][:k]
                predict_label = most_frequent(self.index2label(L_d[n][n_0][:k], range_value, n))
                predict_list.append(predict_label)
            predict_all_list.append(predict_list)
        return predict_all_list

    @staticmethod
    def cal_a(correct_label, predict_list):
        correct = 0
        for i in range(len(correct_label)):
            correct = correct + 1 if correct_label[i] == predict_list[i] else correct
        return correct

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

    def get_dict(self, range_value, attribute):
        _dict = {range_value: {}}
        for idx, item in enumerate(self.data[range_value]):
            _dict[range_value][idx] = []
            for chose in item:
                d = chose[attribute]
                _dict[range_value][idx].append(chose[attribute])
        return _dict

    def get_pos_dict(self, range_value):
        return self.get_dict(range_value, 3)

    def get_label_dict(self, range_value):
        return self.get_dict(range_value, 2)

    def get_n_neighbors_dict(self, range_value):
        vote_dict = {range_value: []}
        for t0 in self.data[range_value]:
            votes = []
            # set the time_series to compute its distance matrix
            self.set_time(t0)
            self.cal_distance_matrix()

            distance_list = self.distance_matrix2distance_list(t0)
            # group (t1_id, t2_id, distance)
            euc_segment = [list(group) for _, group in groupby(distance_list, itemgetter(0))]
            for i in range(len(euc_segment)):
                euc_segment[i] = sorted(euc_segment[i], key=itemgetter(2))
                votes.append(euc_segment[i][:self.k])
            vote_dict[range_value].append(votes)
        return vote_dict

    def distance_matrix2distance_list(self, t0):
        distance_list = []
        for t1 in t0:
            t1_id = t1[0]
            for t2 in t0:
                t2_id = t2[0]
                if t2_id != t1_id:
                    distance = self.distance_matrix[t1_id][t2_id]
                    if distance != 0:
                        distance_list.append((t1_id, t2_id, distance))
        return distance_list

    def set_time(self, t0):
        self.time = []
        for d in t0:
            if isinstance(d, tuple):
                self.time.append(d[1])
            if isinstance(d, list):
                for ts in d:
                    self.time.append(ts[1])

    def cal_distance_matrix(self):
        self.distance_matrix = []
        if load_history_matrix and \
                os.path.exists(MATRIX_FILE):
            self.distance_matrix = load_matrix_from_file(MATRIX_FILE, len(self.poses))
        else:
            if self.dist == 'dtw':
                # Compute DTW distance matrix
                dtw_tools = FastDtw(self.time, self.time, None)
                self.distance_matrix = dtw_tools.dist_matrix()
                self.distance_matrix.tofile(MATRIX_FILE)
            elif self.dist == 'euclidean':
                # euclidean distance matrix
                self.distance_matrix = self.cal_euclidean(self.time)
            if OUTPUT_POS:
                list2file(POS_FILE, self.poses)

    @staticmethod
    def cal_euclidean(times_series):
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
