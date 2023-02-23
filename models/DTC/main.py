import os
import csv
from time import time
from keras.models import Model
from keras.layers import Dense, Reshape, UpSampling2D, Conv2DTranspose, GlobalAveragePooling1D, Softmax, ZeroPadding1D
from keras.losses import kullback_leibler_divergence
import keras.backend as K
from sklearn.cluster import AgglomerativeClustering, KMeans
from .TSClusteringLayer import TSClusteringLayer
from .TAE import temporal_autoencoder
from .tsdistances import *
import numpy as np
import tensorflow as tf


class DTC:
    """
    Deep Temporal Clustering (DTC) model
    # Arguments
        n_clusters: number of clusters
        input_dim: input dimensionality
        timesteps: length of input sequences (can be None for variable length)
        n_filters: number of filters in convolutional layer
        kernel_size: size of kernel in convolutional layer
        strides: strides in convolutional layer
        pool_size: pooling size in max pooling layer, must divide the time series length
        n_units: numbers of units in the two BiLSTM layers
        alpha: coefficient in Student's kernel
        dist_metric: distance metric between latent sequences
        cluster_init: cluster initialization method
    """

    def __init__(self, n_clusters, input_dim, timesteps,
                 n_filters=50, kernel_size=10, strides=1, pool_size=10, n_units=[50, 1],
                 alpha=1.0, dist_metric='eucl', cluster_init='kmeans', heatmap=False):

        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.n_units = n_units
        self.latent_shape = (self.timesteps // self.pool_size, self.n_units[1])
        self.alpha = alpha
        self.dist_metric = dist_metric
        self.cluster_init = cluster_init
        self.heatmap = heatmap
        self.pretrained = False
        self.model = self.autoencoder = self.encoder = self.decoder = None
        if self.heatmap:
            self.heatmap_model = None
            self.heatmap_loss_weight = None
            self.initial_heatmap_loss_weight = None
            self.final_heatmap_loss_weight = None
            self.finetune_heatmap_at_epoch = None

    def initialize(self):
        """
        Create DTC model
        """
        # Create AE models
        self.autoencoder, self.encoder, self.decoder = temporal_autoencoder(input_dim=self.input_dim,
                                                                            timesteps=self.timesteps,
                                                                            n_filters=self.n_filters,
                                                                            kernel_size=self.kernel_size,
                                                                            strides=self.strides,
                                                                            pool_size=self.pool_size,
                                                                            n_units=self.n_units)
        clustering_layer = TSClusteringLayer(self.n_clusters,
                                             alpha=self.alpha,
                                             dist_metric=self.dist_metric,
                                             name='TSClustering')(self.encoder.output)

        # Heatmap-generating network
        if self.heatmap:
            n_heatmap_filters = self.n_clusters  # one heatmap (class activation map) per cluster
            encoded = self.encoder.output
            heatmap_layer = Reshape((-1, 1, self.n_units[1]))(encoded)
            heatmap_layer = UpSampling2D((self.pool_size, 1))(heatmap_layer)
            heatmap_layer = Conv2DTranspose(n_heatmap_filters, (self.kernel_size, 1), padding='same')(heatmap_layer)
            # The next one is the heatmap layer we will visualize
            heatmap_layer = Reshape((-1, n_heatmap_filters), name='Heatmap')(heatmap_layer)
            heatmap_output_layer = GlobalAveragePooling1D()(heatmap_layer)
            # A dense layer must be added only if `n_heatmap_filters` is different from `n_clusters`
            # heatmap_output_layer = Dense(self.n_clusters, activation='relu')(heatmap_output_layer)
            heatmap_output_layer = Softmax()(heatmap_output_layer)  # normalize activations with softmax

        if self.heatmap:
            # Create DTC model
            self.model = Model(inputs=self.autoencoder.input,
                               outputs=[self.autoencoder.output, clustering_layer, heatmap_output_layer])
            # Create Heatmap model
            self.heatmap_model = Model(inputs=self.autoencoder.input,
                                       outputs=heatmap_layer)
        else:
            # Create DTC model
            self.model = Model(inputs=self.autoencoder.input,
                               outputs=[self.autoencoder.output, clustering_layer])

    @property
    def cluster_centers_(self):
        """
        Returns cluster centers
        """
        return self.model.get_layer(name='TSClustering').get_weights()[0]

    @staticmethod
    def weighted_kld(loss_weight):
        """
        Custom KL-divergence loss with a variable weight parameter
        """
        def loss(y_true, y_pred):
            return loss_weight * kullback_leibler_divergence(y_true, y_pred)
        return loss

    def on_epoch_end(self, epoch):
        """
        Update heatmap loss weight on epoch end
        """
        if epoch > self.finetune_heatmap_at_epoch:
            K.set_value(self.heatmap_loss_weight, self.final_heatmap_loss_weight)

    def compile(self, gamma, optimizer, initial_heatmap_loss_weight=None, final_heatmap_loss_weight=None):
        """
        Compile DTC model
        # Arguments
            gamma: coefficient of TS clustering loss
            optimizer: optimization algorithm
            initial_heatmap_loss_weight (optional): initial weight of heatmap loss vs clustering loss
            final_heatmap_loss_weight (optional): final weight of heatmap loss vs clustering loss (heatmap finetuning)
        """
        if self.heatmap:
            self.initial_heatmap_loss_weight = initial_heatmap_loss_weight
            self.final_heatmap_loss_weight = final_heatmap_loss_weight
            self.heatmap_loss_weight = K.variable(self.initial_heatmap_loss_weight)
            self.model.compile(loss=['mse', DTC.weighted_kld(1.0 - self.heatmap_loss_weight), DTC.weighted_kld(self.heatmap_loss_weight)],
                               loss_weights=[1.0, gamma, gamma],
                               optimizer=optimizer)
        else:
            self.model.compile(loss=['mse', 'kld'],
                               loss_weights=[1.0, gamma],
                               optimizer=optimizer)

    def load_weights(self, weights_path):
        """
        Load pre-trained weights of DTC model
        # Arguments
            weight_path: path to weights file (.h5)
        """
        self.model.load_weights(weights_path)
        self.pretrained = True

    def load_ae_weights(self, ae_weights_path):
        """
        Load pre-trained weights of AE
        # Arguments
            ae_weight_path: path to weights file (.h5)
        """
        self.autoencoder.load_weights(ae_weights_path)
        self.pretrained = True

    def dist(self, x1, x2):
        """
        Compute distance between two multivariate time series using chosen distance metric
        # Arguments
            x1: first input (np array)
            x2: second input (np array)
        # Return
            distance
        """
        if self.dist_metric == 'eucl':
            return tsdistances.eucl(x1, x2)
        elif self.dist_metric == 'cid':
            return tsdistances.cid(x1, x2)
        elif self.dist_metric == 'cor':
            return tsdistances.cor(x1, x2)
        elif self.dist_metric == 'acf':
            return tsdistances.acf(x1, x2)
        else:
            raise ValueError('Available distances are eucl, cid, cor and acf!')

    def init_cluster_weights(self, X):
        """
        Initialize with complete-linkage hierarchical clustering or k-means.
        # Arguments
            X: numpy array containing training set or batch
        """
        assert(self.cluster_init in ['hierarchical', 'kmeans'])
        print('Initializing cluster...')

        features = self.encode(X)

        if self.cluster_init == 'hierarchical':
            if self.dist_metric == 'eucl':  # use AgglomerativeClustering off-the-shelf
                hc = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             affinity='euclidean',
                                             linkage='complete').fit(features.reshape(features.shape[0], -1))
            else:  # compute distance matrix using dist
                d = np.zeros((features.shape[0], features.shape[0]))
                for i in range(features.shape[0]):
                    for j in range(i):
                        d[i, j] = d[j, i] = self.dist(features[i], features[j])
                hc = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             affinity='precomputed',
                                             linkage='complete').fit(d)
            # compute centroid
            cluster_centers = np.array([features[hc.labels_ == c].mean(axis=0) for c in range(self.n_clusters)])
        elif self.cluster_init == 'kmeans':
            # fit k-means on flattened features
            km = KMeans(n_clusters=self.n_clusters, n_init=10).fit(features.reshape(features.shape[0], -1))
            cluster_centers = km.cluster_centers_.reshape(self.n_clusters, features.shape[1], features.shape[2])

        self.model.get_layer(name='TSClustering').set_weights([cluster_centers])
        print('Done!')

    def encode(self, x):
        """
        Encoding function. Extract latent features from hidden layer
        # Arguments
            x: data point
        # Return
            encoded (latent) data point
        """
        return self.encoder.predict(x)

    def decode(self, x):
        """
        Decoding function. Decodes encoded sequence from latent space.
        # Arguments
            x: encoded (latent) data point
        # Return
            decoded data point
        """
        return self.decoder.predict(x)

    def predict(self, x):
        """
        Predict cluster assignment.
        """
        q = self.model.predict(x, verbose=0)[1]
        return q.argmax(axis=1)

    @staticmethod
    def target_distribution(q):  # target distribution p which enhances the discrimination of soft label q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def predict_heatmap(self, x):
        """
        Produces TS clustering heatmap from input sequence.
        # Arguments
            x: data point
        # Return
            heatmap
        """
        return self.heatmap_model.predict(x, verbose=0)

    def pretrain(self, X,
                 optimizer='adam',
                 epochs=10,
                 batch_size=64,
                 save_dir='results/tmp',
                 verbose=1):
        """
        Pre-train the autoencoder using only MSE reconstruction loss
        Saves weights in h5 format.
        # Arguments
            X: training set
            optimizer: optimization algorithm
            epochs: number of pre-training epochs
            batch_size: training batch size
            save_dir: path to existing directory where weights will be saved
        """
        print('Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        # Begin pretraining
        t0 = time()
        self.autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs, verbose=verbose)
        print('Pretraining time: ', time() - t0)
        #self.autoencoder.save_weights('{}/ae_weights-epoch{}.h5'.format(save_dir, epochs))
        print('Pretrained weights are saved to {}/ae_weights-epoch{}.h5'.format(save_dir, epochs))
        self.pretrained = True

    def fit(self, X_train, y_train=None,
            X_val=None, y_val=None,
            epochs=100,
            eval_epochs=10,
            save_epochs=10,
            batch_size=64,
            tol=0.001,
            patience=5,
            finetune_heatmap_at_epoch=8,
            save_dir='results/tmp'):
        """
        Training procedure
        # Arguments
           X_train: training set
           y_train: (optional) training labels
           X_val: (optional) validation set
           y_val: (optional) validation labels
           epochs: number of training epochs
           eval_epochs: evaluate metrics on train/val set every eval_epochs epochs
           save_epochs: save model weights every save_epochs epochs
           batch_size: training batch size
           tol: tolerance for stopping criterion
           patience: patience for stopping criterion
           finetune_heatmap_at_epoch: epoch number where heatmap finetuning will start. Heatmap loss weight will
                                      switch from `self.initial_heatmap_loss_weight` to `self.final_heatmap_loss_weight`
           save_dir: path to existing directory where weights and logs are saved
        """
        if not self.pretrained:
            print('Autoencoder was not pre-trained!')

        if self.heatmap:
            self.finetune_heatmap_at_epoch = finetune_heatmap_at_epoch

        # Logging file
        #logfile = open(save_dir + '/dtc_log.csv', 'w')
        fieldnames = ['epoch', 'T', 'L', 'Lr', 'Lc']
        if X_val is not None:
            fieldnames += ['L_val', 'Lr_val', 'Lc_val']
        if y_train is not None:
            fieldnames += ['acc', 'pur', 'nmi', 'ari']
        if y_val is not None:
            fieldnames += ['acc_val', 'pur_val', 'nmi_val', 'ari_val']
        #logwriter = csv.DictWriter(logfile, fieldnames)
        #logwriter.writeheader()

        y_pred_last = None
        patience_cnt = 0

        print('Training for {} epochs.\nEvaluating every {} and saving model every {} epochs.'.format(epochs, eval_epochs, save_epochs))

        for epoch in range(epochs):

            # Compute cluster assignments for training set
            q = self.model.predict(X_train)[1]
            p = DTC.target_distribution(q)

            # Evaluate losses and metrics on training set
            if epoch % eval_epochs == 0:

                # Initialize log dictionary
                logdict = dict(epoch=epoch)

                y_pred = q.argmax(axis=1)
                if X_val is not None:
                    q_val = self.model.predict(X_val)[1]
                    p_val = DTC.target_distribution(q_val)
                    y_val_pred = q_val.argmax(axis=1)

                print('epoch {}'.format(epoch))
                if self.heatmap:
                    loss = self.model.evaluate(X_train, [X_train, p, p], batch_size=batch_size, verbose=False)
                else:
                    loss = self.model.evaluate(X_train, [X_train, p], batch_size=batch_size, verbose=False)
                logdict['L'] = loss[0]
                logdict['Lr'] = loss[1]
                logdict['Lc'] = loss[2]
                print('[Train] - Lr={:f}, Lc={:f} - total loss={:f}'.format(logdict['Lr'], logdict['Lc'], logdict['L']))

                if X_val is not None:
                    val_loss = self.model.evaluate(X_val, [X_val, p_val], batch_size=batch_size, verbose=False)
                    logdict['L_val'] = val_loss[0]
                    logdict['Lr_val'] = val_loss[1]
                    logdict['Lc_val'] = val_loss[2]
                    print('[Val] - Lr={:f}, Lc={:f} - total loss={:f}'.format(logdict['Lr_val'], logdict['Lc_val'], logdict['L_val']))

                # check stop criterion
                if y_pred_last is not None:
                    assignment_changes = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if epoch > 0 and assignment_changes < tol:
                    patience_cnt += 1
                    print('Assignment changes {} < {} tolerance threshold. Patience: {}/{}.'.format(assignment_changes, tol, patience_cnt, patience))
                    if patience_cnt >= patience:
                        print('Reached max patience. Stopping training.')
                        #logfile.close()
                        break
                else:
                    patience_cnt = 0

            # Train for one epoch
            if self.heatmap:
                self.model.fit(X_train, [X_train, p, p], epochs=1, batch_size=batch_size, verbose=False)
                self.on_epoch_end(epoch)
            else:
                self.model.fit(X_train, [X_train, p], epochs=1, batch_size=batch_size, verbose=False)

        # Save the final model
        #logfile.close()
        return y_pred


def dtc(x, y, nclusters):
    x = np.expand_dims(x, axis=2)
    if x.shape[1] % 2 != 0:
        npad = ((0, 0), (0, 1), (0, 0))
        x = np.pad(x, npad, mode='constant', constant_values=0)
        print(x.shape)

    dtc = DTC(n_clusters=nclusters,
              input_dim=x.shape[-1],
              timesteps=x.shape[1],
              n_filters=50,
              kernel_size=10,
              strides=1,
              pool_size=2,
              n_units=[50, 1],
              alpha=1,
              dist_metric='eucl',
              cluster_init='kmeans',
              heatmap=False)

    pretrain_optimizer = 'adam'
    optimizer = 'adam'
    dtc.initialize()
    dtc.model.summary()
    dtc.compile(gamma=1, optimizer=optimizer, initial_heatmap_loss_weight=0.1,
                final_heatmap_loss_weight=0.9)


    dtc.pretrain(X=x, optimizer=pretrain_optimizer,
                     epochs=100, batch_size=128,
                     save_dir='results/tmp')

    # Initialize clusters
    dtc.init_cluster_weights(x)

    # Fit model
    t0 = time()
    dtc.fit(X_train=x, epochs=200, eval_epochs=1, save_epochs=10, batch_size=128, tol=0.001, patience=5, finetune_heatmap_at_epoch=8, save_dir='results/tmp')
    print('Training time: ', (time() - t0))

    # Evaluate
    print('Performance (TRAIN)')
    results = {}
    q = dtc.model.predict(x)[1]
    y_pred = q.argmax(axis=1)

    return y_pred

