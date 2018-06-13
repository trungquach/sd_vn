import numpy as np
from tq.vae import VariationalAutoencoder

import utils, os
import _pickle as cPickle

from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix


def eval(model, data, labels):
    decoded = model.reconstruct(data)
    dist = np.zeros(data.shape[0])

    for i, x in enumerate(data):
        dist[i] = np.linalg.norm(x - decoded[i])

    fpr, tpr, thresholds = roc_curve(labels, dist)

    return fpr, tpr, thresholds, dist

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

def train(vae,mini_batches,test_data,test_label, learning_rate= 0.0001,
          batch_size=100, training_epochs=10, display_steps = 50):
    step = 0
    n_count = 0
    eval_freq = 100
    best_test_score= -np.inf
    #Training cycle
    for epoch in range(training_epochs):
        train_loss = []
        for train_batch in np.random.permutation(mini_batches):
            n_count += 1
            # Fit training using batch data
            loss = vae.partial_fit(train_batch)
            train_loss += [loss]

            if n_count % eval_freq == 0:
                fpr, tpr, thresholds, dist = eval(model=vae, data=test_data, labels=test_label)

                thresold = get_best_threshold_roc(fpr=fpr, tpr=tpr, thresholds=thresholds)
                test_preds = predict_given_threshold(y_scores=dist, threshold=thresold)

                test_score = auc(fpr, tpr)
                f1_test_score = f1_score(y_true=test_label, y_pred=test_preds)

                print('[Test phase] Epoch: %d, roc_auc score: %.3f, f1 score: %.3f' % (epoch + 1, test_score,
                                                                                       f1_test_score))
                if f1_test_score > best_test_score:
                    best_test_score = f1_test_score
                    best_thresold = thresold
                #     best_model = vae

        print('[Train phase] Epoch: %d, average training loss: %.8f' % (epoch + 1, np.mean(train_loss)))
    print('Best F1 score: %.3f' % best_test_score)

    return best_test_score,best_thresold


if __name__ == '__main__':
    # """
    # Create folder if it does not existed
    # """
    # save_out_folder = './saved_model/vae'  # format should be vae_#date
    # if not os.path.exists(save_out_folder):
    #     os.makedirs(save_out_folder)

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
    VAE model
    """
    network_architecture = \
        dict(n_hidden_recog_1=500,  # 1st layer encoder neurons
             n_hidden_recog_2=500,  # 2nd layer encoder neurons
             n_hidden_gener_1=500,  # 1st layer decoder neurons
             n_hidden_gener_2=500,  # 2nd layer decoder neurons
             n_input=datas.shape[1],
             n_z=20)  # dimensionality of latent space
    learning_rate = 0.001

    vae = VariationalAutoencoder(network_architecture, learning_rate=learning_rate, batch_size=batch_size)

    """
     Mini-batchs & perform MinMaxScaler
    """
    vae.build_normalize(train_data=train_datas)  # 1
    norm_datas = vae.transform_raw_data(raw_data=train_datas)
    test_norm_datas = vae.transform_raw_data(raw_data=test_datas)

    mini_batchs = [norm_datas[i:min(i + batch_size, len(norm_datas))] for i in
                   range(0, len(norm_datas), batch_size)]


    """
    Training
    """
    _, threshold = train(vae,mini_batchs,test_norm_datas,test_labels, training_epochs=75)

    """
    Restore model
    """
    # vae.restore(restore_path=save_out_model)  # load best model
    out_test = np.vstack([normal_datas_out, bearing_datas_out, gear_datas_out])
    out_norm_test = vae.transform_raw_data(raw_data=out_test)

    out_test_labels = np.hstack([np.zeros(normal_datas_out.shape[0]),  # 0 for inlier, 1 for outlier
                                 np.ones(bearing_datas_out.shape[0]),
                                 np.ones(gear_datas_out.shape[0])])

    fpr, tpr, thresholds, dist = eval(model=vae, data=out_norm_test, labels=out_test_labels)

    test_score = auc(fpr, tpr)
    out_test_preds = predict_given_threshold(y_scores=dist, threshold=threshold)

    print('[Test out data] roc_auc score: %.3f, f1 score: %.3f' % (test_score, f1_score(y_true=out_test_labels,
                                                                                        y_pred=out_test_preds)))

    print("[Test out data] confusion matrix: \n", confusion_matrix(out_test_labels, out_test_preds))

    pass