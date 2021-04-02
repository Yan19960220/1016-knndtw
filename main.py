from __future__ import absolute_import, division
import matplotlib
import numpy as np
from data import getData
import matplotlib.pyplot as plt
from knndtw import KnnDtw
matplotlib.use('TkAgg')


if __name__ == "__main__":

    k_range = 50
    accuracy, k_list = [], []
    range_list = [50, 100, 200, 300]
    duration_list = np.arange(0.5, 2.5, 0.5).tolist()
    sample_dict = getData(random_range=range_list,
                          DURATION_TO_EXAMINE=duration_list)

    classifier = KnnDtw(k_range, sample_dict,
                        random_range=range_list)
    # dicts = classifier.get_pos_dict(50)
    # dicts2 = classifier.get_label_dict(50)
    # dict3 = classifier.get_n_neighbors_dict(50)
    acc = classifier.merge_view()

    # Drawing
    fig = plt.figure(figsize=(12, 4))
    for key in acc:
        plt.plot([k for k in range(1, len(acc[key])+1)], [s for s in acc[key]], lw=1, marker='o', label=str(key))

    plt.legend()
    # _ = plt.plot([k for k in k_list], [s for s in accuracy], lw=1, marker='o', markerfacecolor='blue')
    plt.title('KNN + DTW for the time series')
    plt.ylabel('Accuracy')
    plt.xlabel('K')
    plt.show()
