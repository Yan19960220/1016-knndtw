from __future__ import absolute_import, division
import matplotlib
from data import getData
import matplotlib.pyplot as plt
from knndtw import KnnDtw
matplotlib.use('TkAgg')


if __name__ == "__main__":
    k_range = 50
    accuracy, k_list = [], []
    ts, time, poses = getData()  # ts(indexes, time_series, label, pos)
    classifier = KnnDtw(k_range, ts, time, poses, dist="dtw")
    k_list, accuracy = classifier.result()
    # matrix = cal_distance_matrix(time)
    # votes = get_n_neighbors(k_range, ts, matrix)
    # k_list, accuracy = vote(ts, k_range, votes)

    fig = plt.figure(figsize=(12, 4))
    _ = plt.plot([k for k in k_list], [s for s in accuracy], lw=1)
    plt.title('KNN + DTW for the time series')
    plt.ylabel('Accuracy')
    plt.xlabel('K')
    plt.show()
