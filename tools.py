# -*- coding: utf-8 -*-
# @Time    : 3/23/21 3:33 PM
# @Author  : Yan
# @Site    : 
# @File    : tools.py.py
# @Software: PyCharm


def file2list(path):
    with open(path, "r") as f:
        lines = [int(line.strip()) for line in f]
    return lines


def list2file(path, list_a):
    with open(path, "w") as f:
        for item in list_a:
            f.write("%s\n" %str(item))