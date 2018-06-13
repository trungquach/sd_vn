import tensorflow as tf
import numpy as np
from nn.vae import VAE
import utils, os
import _pickle as cPickle

from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def eval(model, data, labels):
    decoded = model.get_decoded_output(datas=data)
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

def train(vae, mini_batchs, valida_data, valida_label, save_out_model, n_epoch=150):
    lr_decay = 0.97
    lr_freq = 200
    eval_freq = 100
    n_count = 0
    best_test_score = -np.inf
    best_thresold = -np.inf
    init_lr = vae.init_lr

    for epoch in range(n_epoch):
        train_loss = []

        for train_batch in np.random.permutation(mini_batchs):
            n_count += 1

            if n_count % lr_freq == 0:
                init_lr = init_lr * lr_decay
                print('[Learning rate] Decay to value: ', init_lr)

            loss = vae.batch_train(batch_data=train_batch, lr=init_lr)
            train_loss += [loss]

            if n_count % eval_freq == 0:
                fpr, tpr, thresholds, dist = eval(model=vae, data=valida_data, labels=valida_label)

                thresold = get_best_threshold_roc(fpr=fpr,tpr=tpr,thresholds=thresholds)
                test_preds = predict_given_threshold(y_scores=dist, threshold=thresold)

                test_score = auc(fpr, tpr)
                f1_test_score = f1_score(y_true=valida_label, y_pred=test_preds)

                print('[Test phase] Epoch: %d, roc_auc score: %.3f, f1 score: %.3f' % (epoch + 1, test_score,
                                                                                       f1_test_score))

                if f1_test_score > best_test_score:
                    best_test_score = f1_test_score
                    best_thresold = thresold
                    if save_out_model is not None: vae.save(save_path=save_out_model)

        print('[Train phase] Epoch: %d, average training loss: %.8f' % (epoch + 1, np.mean(train_loss)))
    print('Best F1 score: %.3f' % best_test_score)

    return best_test_score,best_thresold

if __name__ == '__main__':
    """
    Create folder if it does not existed
    """
    save_out_folder = './saved_model/vae'  # format should be vae_#date
    if not os.path.exists(save_out_folder):
        os.makedirs(save_out_folder)

    """
    Data
    """
    split_N = 1024 * 2
    batch_size = 100

    normal_accs = utils.merageVm2012Data(utils.normal_position1_data_path + '/' + utils.csv_file_regular_expression)
    bearing_accs = utils.merageVm2012Data(utils.bearing_position1_data_path + '/' + utils.csv_file_regular_expression)
    gear_accs = utils.merageVm2012Data(utils.gear_position1_data_path + '/' + utils.csv_file_regular_expression)

    normal_datas = utils.spliteAcc2fft(accs=normal_accs, splite_n=split_N, fs=utils.freq)
    bearing_datas = utils.spliteAcc2fft(accs=bearing_accs, splite_n=split_N, fs=utils.freq)
    gear_datas = utils.spliteAcc2fft(accs=gear_accs, splite_n=split_N, fs=utils.freq)

    n_sample_out = 200
    normal_datas, normal_datas_out = normal_datas[n_sample_out:] , normal_datas[:n_sample_out]
    bearing_datas, bearing_datas_out = bearing_datas[n_sample_out:], bearing_datas[:n_sample_out]
    gear_datas, gear_datas_out = gear_datas[n_sample_out:], gear_datas[:n_sample_out]

    datas  = np.vstack([normal_datas,bearing_datas,gear_datas])
    labels = np.hstack([np.zeros(normal_datas.shape[0]), # 0 for inlier, 1 for outlier
                        np.ones(bearing_datas.shape[0]),
                        np.ones(gear_datas.shape[0])])

    train_datas, valida_datas, train_labels, valida_labels = utils.split_train_test(datas=datas, labels=labels, frac=0.8)

    """
    VAE model
    """
    input_dim = datas.shape[1]
    enc_hid_dim = 200
    n_enc_layer = 2
    latent_dim = 100
    dec_hid_dim = 200
    n_dec_layer = 2
    init_lr = 0.001
    n_sample = 2 # not used yet
    beta = 0.5
    use_batch_norm = False
    init_keep_prob = 0.8

    vae = VAE(input_dim=input_dim, enc_hid_dim=enc_hid_dim, n_enc_layer=n_enc_layer, latent_dim=latent_dim,
              dec_hid_dim=dec_hid_dim, n_dec_layer=n_dec_layer, init_lr=init_lr, n_sample=n_sample, beta=beta,
              use_batch_norm=use_batch_norm,init_keep_prob=init_keep_prob)

    # save class instance by using cPickle, main purpose is to save parameters too.
    cPickle.dump(vae,open(os.path.join(save_out_folder,'vae_class.pkl'),'wb'))

    vae.build()

    """
    Mini-batchs & perform MinMaxScaler
    """
    vae.build_normalize(train_data=train_datas) #1
    norm_datas = vae.transform_raw_data(raw_data=train_datas)
    valida_norm_datas = vae.transform_raw_data(raw_data=valida_datas)

    mini_batchs = [norm_datas[i:min(i + batch_size, len(norm_datas))] for i in
                   range(0, len(norm_datas), batch_size)]

    # vae.build_normalize(train_data=datas) #2
    # norm_datas = vae.transform_raw_data(raw_data=datas)
    #
    # mini_batchs = [norm_datas[i:min(i + batch_size, len(norm_datas))] for i in
    #                  range(0, len(norm_datas), batch_size)]

    """
    Training
    """
    save_out_model = os.path.join(save_out_folder,'vae_tensor.ckpt')
    _, threshold = train(vae=vae, mini_batchs=mini_batchs, valida_data=valida_norm_datas, valida_label=valida_labels, save_out_model=save_out_model) #1
    #train(vae=vae, mini_batchs=mini_batchs, valida_data=norm_datas, valida_label=labels,save_out_model=save_out_model) #2

    """
    Restore model
    """
    vae.restore(restore_path=save_out_model) # load best model
    out_test = np.vstack([normal_datas_out,bearing_datas_out,gear_datas_out])
    out_norm_test = vae.transform_raw_data(raw_data=out_test)

    out_test_labels = np.hstack([np.zeros(normal_datas_out.shape[0]), # 0 for inlier, 1 for outlier
                        np.ones(bearing_datas_out.shape[0]),
                        np.ones(gear_datas_out.shape[0])])

    fpr, tpr, thresholds, dist = eval(model=vae, data=out_norm_test, labels=out_test_labels)

    test_score = auc(fpr, tpr)
    out_test_preds = predict_given_threshold(y_scores=dist, threshold=threshold)

    print('[Test out data] roc_auc score: %.3f, f1 score: %.3f' % (test_score, f1_score(y_true=out_test_labels,
                                                                                        y_pred=out_test_preds)))

    print ("[Test out data] confusion matrix: \n", confusion_matrix(out_test_labels, out_test_preds))

    pass



