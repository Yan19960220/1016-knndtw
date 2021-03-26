from __future__ import absolute_import, division
import matplotlib
from data import getData
import matplotlib.pyplot as plt
from knndtw import KnnDtw
matplotlib.use('TkAgg')


if __name__ == "__main__":

    k_range = 50
    accuracy, k_list = [], []
    sample_list = getData([50, 100, 200, 300])  # ts(indexes, time_series, label, pos)

    fig = plt.figure(figsize=(12, 4))
    if isinstance(sample_list, list):
        for item in sample_list:
            (range, (ts, time, poses)) = item
            print(range)
            classifier = KnnDtw(k_range, ts, time, poses)
            k_list, accuracy = classifier.result()
            label = str(range)
            plt.plot([k for k in k_list], [s for s in accuracy], lw=1, marker='o', label=label)
    else:
        label, (ts, time, poses) = sample_list
        classifier = KnnDtw(k_range, ts, time, poses)
        k_list, accuracy = classifier.result()
        plt.plot([k for k in k_list], [s for s in accuracy], lw=1, marker='o', label=str(label))
    plt.legend()
    # _ = plt.plot([k for k in k_list], [s for s in accuracy], lw=1, marker='o', markerfacecolor='blue')
    plt.title('KNN + DTW for the time series')
    plt.ylabel('Accuracy')
    plt.xlabel('K')
    plt.show()
