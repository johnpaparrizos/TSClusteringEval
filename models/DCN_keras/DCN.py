import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import metrics
import numpy as np
from time import time
from keras import callbacks
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from sklearn.cluster import KMeans
from keras.legacy import interfaces
from keras.layers import Input, Dense
from keras.optimizers import Optimizer
from keras.engine.topology import Layer
from keras.losses import mean_squared_error
from keras.initializers import VarianceScaling


import tensorflow as tf
from keras import backend as K

num_cores = 1
num_CPU = 1
num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU}
                       )

session = tf.Session(config=config)
K.set_session(session)

def autoencoder_with_kmeans(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    clustering_input = Input(shape=(dims[-1],), name='clustering_input')

    x = Input(shape=(dims[0],), name='input')
    h = x

    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here
    y = h
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)
    return Model(inputs=[x, clustering_input], outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder'), clustering_input
    

def loss_wrapper(encoded_X,label_centers,lambd):
    def loss(y_true, y_pred):
        cost_clustering = K.mean(K.square(label_centers-encoded_X),axis=-1)
        cost_reconstruction = K.mean(K.square(y_true-y_pred),axis=-1)
        cost = lambd*cost_clustering+cost_reconstruction
        return cost
    return loss

class DCN(object):
    def __init__(self, dims, n_clusters, lambd=0.5, init='glorot_uniform'):
        super(DCN, self).__init__()
       
        self.dims = dims
        print(dims)
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.lambd = lambd
        self.autoencoder, self.encoder, self.clustering_input = autoencoder_with_kmeans(self.dims, init=init)

        self.centers = np.zeros((self.n_clusters, self.dims[-1]))
        self.count = 100*np.ones(self.n_clusters, dtype=np.int)
    
    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='./models/DCN_keras'):
        print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs/10) != 0 and epoch % int(epochs/10) != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(
                                              'encoder_%d' % (int(len(self.model.layers) / 2) - 1)).output)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=1)
                    y_pred = km.fit_predict(features)

            cb.append(PrintACC(x, y))

        # begin pretraining
        t0 = time()
        temp = []
        for i in range(x.shape[0]):
            t = []
            for k in range(self.dims[-1]):
                t.append(0)
            temp.append(t)
        temp = np.array(temp)
        self.autoencoder.fit([x, temp], x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        #self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        self.pretrained = True
            
    def init_centers(self,x, y=None):
        #from tslearn.clustering import TimeSeriesKMeans
        #kmeans = TimeSeriesKMeans(n_clusters=self.n_clusters, metric="euclidean", max_iter=300, init='random', n_init=1)
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(self.encoder.predict(x))
        self.all_pred = kmeans.labels_
        self.centers = kmeans.cluster_centers_
        
    def compile(self):
        self.autoencoder.compile(optimizer='adam', loss=loss_wrapper(self.encoder.output,self.clustering_input,0.5))

    def fit(self, x, y, epoches, batch_size=256, save_dir='./models/DCN_keras'):                    
        m = x.shape[0]
        self.count = 100*np.ones(self.n_clusters, dtype=np.int)
        best_inertia = float("inf")
        for step in range(epoches):   
            cost = [] # all cost
            
            if m % batch_size == 0:
                total_batches = int(m/batch_size)
            else:
                total_batches = int(m/batch_size) + 1

            for batch_index in range(total_batches):
                X_batch = x[batch_index*batch_size:(batch_index+1)*batch_size,:]
                labels_of_centers = self.centers[self.all_pred[batch_index*batch_size:(batch_index+1)*batch_size]]

                c1 = self.autoencoder.train_on_batch([X_batch, labels_of_centers], X_batch)
                cost.append(c1)
                
                reductX = self.encoder.predict(X_batch)
                self.all_pred[batch_index*batch_size:(batch_index+1)*batch_size], self.centers, self.count = self.batch_km(reductX, self.centers, self.count)
                
            if step%10 == 0:
                reductX = self.encoder.predict(x)
                km_model = KMeans(self.n_clusters, init=self.centers).fit(reductX)
                #from tslearn.clustering import TimeSeriesKMeans
                #km_model = TimeSeriesKMeans(n_clusters=self.n_clusters, metric="euclidean", max_iter=300, init='random', n_init=1).fit(reductX)
                inertia = km_model.inertia_
                self.all_pred = km_model.labels_
                self.centers = km_model.cluster_centers_
            
                if best_inertia > inertia:
                    best_inertia = inertia
                    best_pred = self.predict(x) 

        #self.autoencoder.save_weights(save_dir + '/DCN_model_final.h5')
        return inertia, best_inertia, best_pred 
        
    def batch_km(self, data, center, count):
        """
        Function to perform a KMeans update on a batch of data, center is the
        centroid from last iteration.
        """
        N = data.shape[0]
        K = center.shape[0]
    
        # update assignment
        idx = np.zeros(N, dtype=np.int)
        for i in range(N):
            dist = np.inf
            ind = 0
            for j in range(K):
                temp_dist = np.linalg.norm(data[i] - center[j])
                if temp_dist < dist:
                    dist = temp_dist
                    ind = j
            idx[i] = ind
    
        # update centriod
        center_new = center
        for i in range(N):
            c = idx[i]
            count[c] += 1
            eta = 1.0/count[c]
            center_new[c] = (1 - eta) * center_new[c] + eta * data[i]
        center_new.astype(np.float32)
        return idx, center_new, count

    def get_centers_and_types_of_points(self,reductX):
        distances = np.abs(reductX - self.centers[:, np.newaxis])
        label_types = np.min(np.argmin(distances, axis=0),axis=1)
        labels_of_centers = self.centers[label_types]
        return labels_of_centers, label_types
        
    def load_weights(self, weights):  # load weights of DEC model
        self.autoencoder.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        reductX = self.encoder.predict(x)
        labels_of_centers, label_types=self.get_centers_and_types_of_points(reductX)
        return label_types


def dcn_clustering(ts, labels, nclusters, params, best=True):
    architecture = params[0]
    dcn = DCN(dims=[ts.shape[-1]] + architecture.tolist(), n_clusters=nclusters, lambd=0.05)
    dcn.compile()

    pretrain_epochs = 50

    dcn.pretrain(x=ts, epochs=pretrain_epochs)
    dcn.init_centers(ts, labels)

    inertia, best_inertia, best_y_pred = dcn.fit(ts, labels, epoches=50, batch_size=8)
    y_pred = dcn.predict(ts)

    if best:
        return best_inertia, best_y_pred
    else:
        return inertia, y_pred
