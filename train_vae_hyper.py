from sklearn.neighbors.base import NeighborsBase, KNeighborsMixin, UnsupervisedMixin
from nn.vae import VAE
import tensorflow as tf
import train_vae
import json
import utils,os,_pickle as cPickle
import numpy as np
from collections import defaultdict
from sklearn.metrics import auc, confusion_matrix, f1_score

from shutil import rmtree # for remove folder
from distutils.dir_util import copy_tree # for copy

batch_size = 100
save_vae_hyper_folder = './saved_model/vae_hyper' # format should be vae_hyper_#date
global_best_test_score = -np.inf

"""
Create folder if it does not existed
"""
if not os.path.exists(save_vae_hyper_folder):
    os.makedirs(save_vae_hyper_folder)

def copy(src,dest):
    if os.path.isfile(dest):
        rmtree(dest)

    copy_tree(src, dest)

class HyperVAE(NeighborsBase, KNeighborsMixin, UnsupervisedMixin):
    normal_datas = None
    abnormal_datas = None

    def __init__(self, vae_params, normal_datas, abnormal_datas):

        self.vae_params = vae_params
        if HyperVAE.abnormal_datas is None:
            HyperVAE.normal_datas = normal_datas
            HyperVAE.abnormal_datas = abnormal_datas

        """
        Save path
        """
        self.save_best_folder = os.path.join(save_vae_hyper_folder, 'best')
        self.save_candidate_folder = os.path.join(save_vae_hyper_folder, 'candidate')
        self.save_log = os.path.join(save_vae_hyper_folder,'log.txt')

        if os.path.exists(self.save_log):
            os.remove(self.save_log)

    def set_params(self, **params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            self.vae_params[key] = value

        return self

    # build model with the newer parameters
    def fit(self, X, y=None):
        print ("fit X: ", X.shape)
        print ("fit y: ", y.shape if y is not None else '')

        """
        Split train and test set
        """
        train_datas = self.normal_datas[X]
        valid_datas = np.vstack([np.delete(self.normal_datas,X,axis=0),
                                self.abnormal_datas])
        valid_labels = np.hstack([np.zeros(self.normal_datas.shape[0] - X.shape[0]),
                                  np.ones(self.abnormal_datas.shape[0])])

        """
        Rebuild VAE and train
        """
        global global_best_test_score

        tf.reset_default_graph()

        # rebuild VAE
        vae = VAE(**self.vae_params)
        #cPickle.dump(vae, open(os.path.join(self.save_candidate_folder, 'vae_class.pkl'), 'wb'))

        vae.build()

        """
        Normalization
        """
        vae.build_normalize(train_data=train_datas)
        norm_train_datas = vae.transform_raw_data(raw_data=train_datas)
        norm_valid_datas = vae.transform_raw_data(raw_data=valid_datas)

        """
        Mini Batchs
        """
        mini_batchs = [norm_train_datas[i:min(i + batch_size, len(norm_train_datas))] for i in
                     range(0, len(norm_train_datas), batch_size)]

        """
        Train
        """
        self.best_test_score, _ = train_vae.train(vae=vae, mini_batchs=mini_batchs, valida_data=norm_valid_datas,
                                                  valida_label=valid_labels, save_out_model=None, n_epoch=30)
        # self.best_test_score, _ = train_vae.train(vae=vae, mini_batchs=mini_batchs, test_data=norm_valid_datas,
        #                         test_label=valid_labels,save_out_model=os.path.join(self.save_candidate_folder, 'vae_tensor.ckpt'))

        """
        Save result
        """
        print ("Perform training with the below parameters: ")
        print ("------------------------------------------- ")
        print (json.dumps(self.vae_params,indent=2))
        print ("------------------------------------------- ")
        print ("Result (F1): ", self.best_test_score)

        # # save results to log file
        # with open(self.save_log,'a') as f:
        #     f.write('\n+++++++++++++++++++++++++++++++++++++++++++++++\n')
        #
        #     f.write("\n parameters: %s \n" % json.dumps(self.vae_params,indent=2))
        #     f.write("\n result (auc): %.5f \n" % self.best_test_score)
        #
        #     f.write('\n+++++++++++++++++++++++++++++++++++++++++++++++\n')
        #
        # if self.best_test_score > global_best_test_score:
        #     copy(src=self.save_candidate_folder,dest=self.save_best_folder)

    def score(self, X, y=None):
        return self.best_test_score

if __name__ == "__main__":
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

    print ('Loaded normal_datas, with shape: ', normal_datas.shape)
    print ('Loaded bearing_datas, with shape: ', bearing_datas.shape)
    print ('Loaded gear_datas, with shape: ', gear_datas.shape)

    n_sample_out = 200
    normal_datas, normal_datas_out = normal_datas[n_sample_out:], normal_datas[:n_sample_out]
    bearing_datas, bearing_datas_out = bearing_datas[n_sample_out:], bearing_datas[:n_sample_out]
    gear_datas, gear_datas_out = gear_datas[n_sample_out:], gear_datas[:n_sample_out]

    """
    Specify hyper-parameters
    """
    params = {
        "input_dim": [normal_datas.shape[1]],  # user must specify
        "enc_hid_dim": [200,300],
        "n_enc_layer": [2],
        "latent_dim": [100],
        "dec_hid_dim": [200,300],
        "n_dec_layer": [2],
        "init_lr": [0.001],
        "n_sample": [2],  # not used yet
        "beta": [0.5,0,1],
        "use_batch_norm":[False,True],
        "init_keep_prob": [0.8]
    }

    """
    VAE model
    """
    init_param = {k:v[0] for k,v in params.items()}
    vae_hyper = HyperVAE(vae_params=init_param, normal_datas=normal_datas,
                         abnormal_datas=np.vstack([bearing_datas, gear_datas]))

    """
    Apply RandomSearchCV to search for best parameters
    """
    from sklearn.model_selection import RandomizedSearchCV

    rs = RandomizedSearchCV(estimator=vae_hyper, param_distributions=params, n_jobs=1, verbose=2, n_iter=10,
                            cv=5, return_train_score=False, refit=False)
    rs.fit(X=np.arange(normal_datas.shape[0]),y=None)

    print ("Best parameters found: ")
    print (json.dumps(rs.best_params_, indent=2))

    """
    Create new VAE model for whole dataset with best parameter found
    """
    tf.reset_default_graph()
    vae = VAE(**rs.best_params_)

    # save class instance by using cPickle, main purpose is to save parameters too.
    cPickle.dump(vae, open(os.path.join(save_vae_hyper_folder, 'vae_class.pkl'), 'wb'))
    vae.build()

    """
    Prepare data
    """
    datas = np.vstack([normal_datas, bearing_datas, gear_datas])
    labels = np.hstack([np.zeros(normal_datas.shape[0]),  # 0 for inlier, 1 for outlier
                        np.ones(bearing_datas.shape[0]),
                        np.ones(gear_datas.shape[0])])

    train_datas, test_datas, train_labels, test_labels = utils.split_train_test(datas=datas, labels=labels, frac=0.8)

    """
    Mini-batchs & perform MinMaxScaler
    """
    vae.build_normalize(train_data=train_datas)  # 1
    norm_datas = vae.transform_raw_data(raw_data=train_datas)
    test_norm_datas = vae.transform_raw_data(raw_data=test_datas)

    mini_batchs = [norm_datas[i:min(i + batch_size, len(norm_datas))] for i in
                   range(0, len(norm_datas), batch_size)]


    """
    Train
    """
    save_out_model = os.path.join(save_vae_hyper_folder, 'vae_tensor.ckpt')
    _, threshold = train_vae.train(vae=vae, mini_batchs=mini_batchs, valida_data=test_norm_datas, valida_label=test_labels,
                                   save_out_model=save_out_model, n_epoch=200)  # 1

    """
    Testing
    """
    vae.restore(restore_path=save_out_model)  # load best model
    out_test = np.vstack([normal_datas_out, bearing_datas_out, gear_datas_out])
    out_norm_test = vae.transform_raw_data(raw_data=out_test)

    out_test_labels = np.hstack([np.zeros(normal_datas_out.shape[0]),  # 0 for inlier, 1 for outlier
                                 np.ones(bearing_datas_out.shape[0]),
                                 np.ones(gear_datas_out.shape[0])])

    fpr, tpr, thresholds, dist = train_vae.eval(model=vae, data=out_norm_test, labels=out_test_labels)

    test_score = auc(fpr, tpr)
    out_test_preds = train_vae.predict_given_threshold(y_scores=dist, threshold=threshold)

    print('[Test out data] roc_auc score: %.3f, f1 score: %.3f' % (test_score, f1_score(y_true=out_test_labels,
                                                                                        y_pred=out_test_preds)))

    print("[Test out data] confusion matrix: \n", confusion_matrix(out_test_labels, out_test_preds))

    pass