import tensorflow as tf
import numpy as np
from nn.vae import VAE
import utils

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def eval(model, data, labels):
    decoded = model.get_decoded_output(datas=data)
    dist = np.zeros(data.shape[0])

    for i, x in enumerate(data):
        dist[i] = np.linalg.norm(x - decoded[i])

    fpr, tpr, thresholds = roc_curve(labels, dist)
    roc_auc = auc(fpr, tpr)

    return roc_auc

if __name__ == '__main__':
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


    datas  = np.vstack([normal_datas,bearing_datas,gear_datas])
    labels = np.hstack([np.zeros(normal_datas.shape[0]), # 0 for inlier, 1 for outlier
                        np.ones(bearing_datas.shape[0]),
                        np.ones(gear_datas.shape[0])])

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
    n_sample = 2

    vae = VAE(input_dim=input_dim, enc_hid_dim=enc_hid_dim, n_enc_layer=n_enc_layer, latent_dim=latent_dim,
              dec_hid_dim=dec_hid_dim, n_dec_layer=n_dec_layer, init_lr=init_lr, n_sample=n_sample, beta=0.0)
    vae.build()

    """
    Mini-batchs & perform MinMaxScaler
    """
    vae.build_normalize(train_data=datas)
    norm_datas = vae.transform_raw_data(raw_data=datas)

    mini_batchs = [norm_datas[i:min(i + batch_size, len(norm_datas))] for i in
                     range(0, len(norm_datas), batch_size)]

    """
    Training
    """
    n_epoch = 200
    lr_decay = 0.98
    lr_freq = 200
    eval_freq = 100
    n_count = 0
    best_test_score = -np.inf

    for epoch in range(n_epoch):
        train_loss = []

        for train_batch in np.random.permutation(mini_batchs):
            n_count += 1

            if n_count % lr_freq == 0:
                init_lr = init_lr * lr_decay
                print ('Decay lr: ', init_lr )

            loss = vae.batch_train(batch_data=train_batch, lr = init_lr)
            train_loss += [loss]

            if n_count % eval_freq == 0:
                test_score = eval(model=vae,data=norm_datas,labels=labels)
                print ('[Test phase] Epoch: %d, roc_auc score: %.3f' % (epoch + 1, test_score))

                if test_score > best_test_score: best_test_score = test_score

                # saved model here (if necessary)

        print ('[Train phase] Epoch: %d, average training loss: %.8f' % (epoch + 1, np.mean(train_loss)))
    print ('Best ROC_AUC score: %.3f' % best_test_score )

