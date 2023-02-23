from time import time
import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
from sklearn import metrics
from .DEC import cluster_acc, ClusteringLayer, autoencoder


class IDEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0):
        super(IDEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder = autoencoder(self.dims)
        hidden = self.autoencoder.get_layer(name='encoder_%d' % (self.n_stacks - 1)).output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

        # prepare IDEC model
        clustering_layer = ClusteringLayer(self.n_clusters, alpha=self.alpha, name='clustering')(hidden)
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[clustering_layer, self.autoencoder.output])
        hidden2 = self.model.get_layer(name='encoder_%d' % (self.n_stacks - 1)).output
        self.encoder2 = Model(inputs=self.model.input, outputs=hidden2)

        self.pretrained = False
        self.centers = []
        self.y_pred = []
        self.best_y_pred = []
        self.best_loss = float("inf")
        self.best_inertia = None
        self.loss = None

    def pretrain(self, x, batch_size=256, epochs=200, optimizer='adam'):
        from keras.optimizers import Adam
        self.autoencoder.compile(loss='mse', optimizer=Adam(lr=0.0001))  # SGD(lr=0.01, momentum=0.9),
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs)
        #self.autoencoder.save_weights('./models/IDEC/ae_weights.h5')
        self.pretrained = True

    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam'):
        from keras.optimizers import Adam
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=Adam(lr=0.0001))

    def fit(self, x, y=None, batch_size=256, maxiter=2e4, tol=1e-3, update_interval=140,
            ae_weights=None, save_dir='./results/idec'):

        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        if not self.pretrained and ae_weights is None:
            self.pretrain(x, batch_size)
            self.pretrained = True
        elif ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=300)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            print('Epoch:', ite)
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(cluster_acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    loss = np.round(loss, 5)

                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    break

            epoch_loss = 0
            for index in range((x.shape[0]//batch_size)+1):
                if (index + 1) * batch_size > x.shape[0]:
                    loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                    y=[p[index * batch_size::], x[index * batch_size::]])
                    #index = 0
                else:
                    loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                    y=[p[index * batch_size:(index + 1) * batch_size],
                                                        x[index * batch_size:(index + 1) * batch_size]])
                    #index += 1
                epoch_loss = epoch_loss + loss[0]

            epoch_loss = epoch_loss / (index+1)

            if self.best_loss > epoch_loss:
                self.best_loss = epoch_loss
                #q, _ = self.model.predict(x, verbose=0)
                #self.y_pred = q.argmax(1)
                self.best_y_pred = self.y_pred
                #h = self.encoder2.predict(x)
                #self.best_inertia = kmeans.fit(h).inertia_

        self.loss = epoch_loss
        q, _ = self.model.predict(x, verbose=0)
        self.y_pred = q.argmax(1)
        h = self.encoder2.predict(x)
        self.inertia = kmeans.fit(h).inertia_
        return self.best_loss, self.best_y_pred, self.best_inertia, self.loss, self.y_pred, self.inertia


def idec_clustering(x, y, n_clusters, params, best=False):
    pretrain_epochs = 100
    batch_size = 256
    tol = 0.001
    gamma = 0.1
    ae_weights = None
    maxiter = 100
    save_dir = './models/IDEC/'
    architecture = params[0]
    optimizer = 'adam'  # SGD(lr=0.01, momentum=0.99)
    update_interval = 10

    idec = IDEC(dims=[x.shape[-1]] + architecture.tolist(), n_clusters=n_clusters)
    idec.model.summary()

    idec.pretrain(x, batch_size=batch_size, epochs=pretrain_epochs, optimizer=optimizer)
    idec.compile(loss=['kld', 'mse'], loss_weights=[gamma, 1], optimizer=optimizer)
    idec.fit(x, y=y, batch_size=batch_size, tol=tol, maxiter=maxiter,
             update_interval=update_interval, ae_weights=ae_weights, save_dir=save_dir)

    if best:
        return idec.best_loss, idec.best_inertia, idec.best_y_pred
    else:
        return idec.loss, idec.inertia, idec.y_pred
