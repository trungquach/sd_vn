import tensorflow as tf
import numpy as np
import utils

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor

if __name__ == '__main__':
    """
    Data
    """
    split_N = 1024 * 2
    batch_size = 100

    normal_accs = utils.merageVm2012Data(utils.normal_position2_data_path + '/' + utils.csv_file_regular_expression)
    bearing_accs = utils.merageVm2012Data(utils.bearing_position2_data_path + '/' + utils.csv_file_regular_expression)
    gear_accs = utils.merageVm2012Data(utils.gear_position2_data_path + '/' + utils.csv_file_regular_expression)


    normal_datas = utils.spliteAcc2fft(accs=normal_accs, splite_n=split_N, fs=utils.freq)
    bearing_datas = utils.spliteAcc2fft(accs=bearing_accs, splite_n=split_N, fs=utils.freq)
    gear_datas = utils.spliteAcc2fft(accs=gear_accs, splite_n=split_N, fs=utils.freq)


    datas  = np.vstack([normal_datas,bearing_datas,gear_datas])
    labels = np.hstack([np.zeros(normal_datas.shape[0]), # 0 for inlier, 1 for outlier
                        np.ones(bearing_datas.shape[0]),
                        np.ones(gear_datas.shape[0])])

    """
    LOF model
    """
    clf = LocalOutlierFactor(n_neighbors=400)
    clf.fit(datas)

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt


    scores = -clf.negative_outlier_factor_
    dist = np.zeros(datas.shape[0])

    for i, x in enumerate(datas):
        dist[i] = np.linalg.norm(x - scores[i])

    fpr, tpr, thresholds = roc_curve(labels, dist)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='red', label='AUC = %0.2f)' % roc_auc)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.title('ROC Autoencoder 100-80-100 ReLU/Sigmoid synth\_multidim\_100\_000')
    plt.legend(loc="lower right")
    plt.show()



