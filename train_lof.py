import tensorflow as tf
import numpy as np
import utils

from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor

def eval(model, data, labels):
    scores = -model._decision_function(data)

    fpr, tpr, thresholds = roc_curve(labels, scores)

    return fpr, tpr, thresholds, scores

def get_best_threshold_roc(fpr,tpr,thresholds):
    # the best one is the point which closest to the top left corner.
    top_left_point = (0.0,1.0) #(FPR, TPR)

    dists = np.array([np.sqrt((0.0 - _fpr)**2 + (1 - _tpr)**2 ) for (_fpr,_tpr) in zip(fpr,tpr)] )
    max_idx = np.argsort(dists)[0]

    return thresholds[max_idx]

def predict_given_threshold(y_scores, threshold):
    # 0 is inlier, 1 is outlier

    y_preds = np.zeros(shape=y_scores.shape[0])
    y_preds[y_scores >= threshold] = 1.0

    return y_preds

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

    n_sample_out = 200
    normal_datas, normal_datas_out = normal_datas[n_sample_out:], normal_datas[:n_sample_out]
    bearing_datas, bearing_datas_out = bearing_datas[n_sample_out:], bearing_datas[:n_sample_out]
    gear_datas, gear_datas_out = gear_datas[n_sample_out:], gear_datas[:n_sample_out]

    datas = np.vstack([normal_datas, bearing_datas, gear_datas])
    labels = np.hstack([np.zeros(normal_datas.shape[0]),  # 0 for inlier, 1 for outlier
                        np.ones(bearing_datas.shape[0]),
                        np.ones(gear_datas.shape[0])])

    train_datas, test_datas, train_labels, test_labels = utils.split_train_test(datas=datas, labels=labels, frac=0.8)

    """
    LOF model
    """
    best_clf = None
    best_test_score = -np.inf
    best_threshold  = -np.inf

    for n_neighbor in [20,40]:
        for n_contamination in [0.05,0.1]:
            # train
            clf = LocalOutlierFactor(n_neighbors=n_neighbor, contamination=n_contamination)
            clf.fit(train_datas)

            # evaluating with validation dataset
            fpr, tpr, thresholds, scores = eval(model = clf, data = test_datas, labels = test_labels)

            thresold = get_best_threshold_roc(fpr=fpr, tpr=tpr, thresholds=thresholds)
            test_preds = predict_given_threshold(y_scores=scores, threshold=thresold)

            test_score = auc(fpr, tpr)
            f1_test_score = f1_score(y_true=test_labels, y_pred=test_preds)

            if f1_test_score >= best_test_score:
                print ("n_neighbor: %d, n_contamination: %f ==> found best test score..." % (n_neighbor, n_contamination))

                best_clf = clf
                best_test_score = f1_test_score
                best_threshold = thresold

            print('n_neighbor: %d, n_contamination: %f, roc_auc score: %.3f, f1 score: %.3f' % (n_neighbor, n_contamination, test_score,
                                                                                   f1_test_score))

    """
    Restore model
    """
    out_test = np.vstack([normal_datas_out, bearing_datas_out, gear_datas_out])
    out_test_labels = np.hstack([np.zeros(normal_datas_out.shape[0]),  # 0 for inlier, 1 for outlier
                                 np.ones(bearing_datas_out.shape[0]),
                                 np.ones(gear_datas_out.shape[0])])

    fpr, tpr, thresholds, dist = eval(model=best_clf, data=out_test, labels=out_test_labels)
    test_score = auc(fpr, tpr)
    out_test_preds = predict_given_threshold(y_scores=dist, threshold=best_threshold)

    print('[Test out data] roc_auc score: %.3f, f1 score: %.3f' % (test_score, f1_score(y_true=out_test_labels,
                                                                                        y_pred=out_test_preds)))

    print("[Test out data] confusion matrix: \n", confusion_matrix(out_test_labels, out_test_preds))

    # from sklearn.metrics import roc_curve, auc
    # import matplotlib.pyplot as plt
    #
    # scores = -clf.negative_outlier_factor_
    # dist = np.zeros(datas.shape[0])
    #
    # for i, x in enumerate(datas):
    #     dist[i] = np.linalg.norm(x - scores[i])
    #
    # fpr, tpr, thresholds = roc_curve(labels, dist)
    # roc_auc = auc(fpr, tpr)
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(fpr, tpr, color='red', label='AUC = %0.2f)' % roc_auc)
    # plt.xlim((0, 1))
    # plt.ylim((0, 1))
    # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # plt.xlabel('False Positive rate')
    # plt.ylabel('True Positive rate')
    # plt.title('ROC Autoencoder 100-80-100 ReLU/Sigmoid synth\_multidim\_100\_000')
    # plt.legend(loc="lower right")
    # plt.show()



