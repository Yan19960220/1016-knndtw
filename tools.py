# -*- coding: utf-8 -*-
# @Time    : 3/23/21 3:33 PM
# @Author  : Yan
# @Site    : 
# @File    : tools.py.py
# @Software: PyCharm
import numpy as np


def file2list(path):
    with open(path, "r") as f:
        lines = [int(line.strip()) for line in f]
    return lines


def list2file(path, list_a):
    with open(path, "w") as f:
        for item in list_a:
            f.write("%s\n" % str(item))


def most_frequent(List):
    return max(set(List), key=List.count)


def most_common(List):
    count = []
    for i in List:
        count.append(most_frequent(i))
    return count


def most_common_row(List):
    common_list = []
    for i in range(np.shape(List)[1]):
        temp = []
        for l in List:
            temp.append(l[i])
        common_list.append(most_frequent(temp))
    return common_list
