import tensorflow as tf
from . import model
from . import data_utils
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import anndata
from . import metrics
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import logging
import itertools
from functools import partial

def guassian_kernel(source, target):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
     source: (b1,n)的X分布样本数组
     target:（b2，n)的Y分布样本数组
    Return:
      kernel_val: 对应的核矩阵
    '''
    # 堆叠两组样本，上面是X分布样本，下面是Y分布样本，得到（b1+b2,n）组总样本
    n_samples = int(source.shape[0])+int(target.shape[0])
    total = np.concatenate((source, target), axis=0)
    # 对总样本变换格式为（1,b1+b2,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按行复制
    total0 = np.expand_dims(total,axis=0)
    total0= np.broadcast_to(total0,[int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    # 对总样本变换格式为（b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按复制
    total1 = np.expand_dims(total,axis=1)
    total1=np.broadcast_to(total1,[int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差
    # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
    L2_distance_square = np.cumsum(np.square(total0-total1),axis=2)
    #调整高斯核函数的sigma值
    bandwidth = np.sum(L2_distance_square) / (n_samples**2-n_samples)
    #高斯核函数的数学表达式
    kernel_val = np.exp(-L2_distance_square / bandwidth)
    #得到最终的核矩阵
    return kernel_val

def MMD(source, target):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
     source: 源域数据，行表示样本数目，列表示样本数据维度
     target: 目标域数据 同source
    Return:
     loss: MMD loss
    '''
    batch_size = int(source.shape[0])#一般默认为X和Y传入的样本的总批次样本数是一致的
    kernels = guassian_kernel(source, target)
    #将核矩阵分成4部分
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    # 这里计算出的n_loss是每个维度上的MMD距离，一般还会做均值化处理
    n_loss= loss / float(batch_size)
    return np.mean(n_loss)

def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

def scale(z):
    zmax = tf.reduce_max(z,axis=1, keepdims=True)
    zmin = tf.reduce_min(z,axis=1, keepdims=True)
    z_std = (z - zmin) / (zmax - zmin)
    return z_std

def l2_activation(x):
    x = scale(x)
    x = tf.nn.l2_normalize(x,axis=1)
    return x

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
    """
    def __init__(self, *args, **kwargs):
        super(BatchNormalization, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        kwargs['training'] = True
        return super(BatchNormalization, self).call(inputs, **kwargs)

class BaseModel(tf.keras.models.Model):
    def __init__(
        self, 
        input_dims,
        latent_dims,
        hidden_dims,
        celltype_dims,
        ae_hidden_layers,
        disc_hidden_layers,
        pred_hidden_layers,
        always_batch_norm,
    ):
        super(BaseModel, self).__init__()
        self.encoder = self.build_encoder(input_dims,latent_dims,hidden_dims,ae_hidden_layers,output_dropout_rate=0.5,always_batch_norm=always_batch_norm)
        self.decoder = self.build_decoder(latent_dims,input_dims,hidden_dims,ae_hidden_layers,output_dropout=False)
        self.disc = self.build_disc(latent_dims,hidden_dims,disc_hidden_layers)
        self.pred = self.build_pred(latent_dims,celltype_dims,hidden_dims,pred_hidden_layers)
        

    def build_encoder(
        self,
        input_dims,
        latent_dims,
        hidden_dims,
        ae_hidden_layers=3,
        output_dropout=True,
        output_dropout_rate=0.5,
        hidden_initializer=tf.keras.initializers.HeNormal(),
        output_initializer=tf.keras.initializers.GlorotUniform(),
        output_activation=l2_activation,
        always_batch_norm=False
    ):
        if always_batch_norm:
            bathnorm = BatchNormalization
        else:
            bathnorm = tf.keras.layers.BatchNormalization
        encoder = tf.keras.Sequential()
        encoder.add(tf.keras.Input(shape=(input_dims,)))
        for i in range(ae_hidden_layers):
            encoder.add(tf.keras.layers.Dense(hidden_dims,kernel_initializer=hidden_initializer,kernel_regularizer='l1_l2'))
            encoder.add(bathnorm(name='encoder_bn'+str(i)))
            encoder.add(tf.keras.layers.LeakyReLU())
        if output_dropout:
            encoder.add(tf.keras.layers.Dropout(output_dropout_rate, name='encoder_dp'))
        encoder.add(tf.keras.layers.Dense(latent_dims, activation=output_activation,kernel_initializer=output_initializer))
        return encoder
    
    def build_decoder(
        self,
        input_dims,
        output_dims,
        hidden_dims,
        ae_hidden_layers=3,
        output_dropout=False,
        output_dropout_rate=0.5,
        hidden_initializer=tf.keras.initializers.HeNormal(),
        output_initializer=tf.keras.initializers.GlorotUniform(),
        output_activation=None
    ):
        decoder = tf.keras.Sequential()
        decoder.add(tf.keras.Input(shape=(input_dims,)))
        for i in range(ae_hidden_layers):
            decoder.add(tf.keras.layers.Dense(hidden_dims,kernel_initializer=hidden_initializer,kernel_regularizer='l1_l2'))
            decoder.add(tf.keras.layers.LeakyReLU())
        if output_dropout:
            decoder.add(tf.keras.layers.Dropout(output_dropout_rate, name='decoder_dp'))
        decoder.add(tf.keras.layers.Dense(output_dims,kernel_initializer=output_initializer))
        return decoder
    
    def build_disc(
        self,
        input_dims,
        hidden_dims,
        disc_hidden_layers=1,
        hidden_initializer=tf.keras.initializers.HeNormal(),
        output_initializer=tf.keras.initializers.GlorotUniform(),
        output_activation='sigmoid'
    ):
        disc = tf.keras.Sequential()
        disc.add(tf.keras.Input(shape=input_dims,))
        for i in range(disc_hidden_layers):
            disc.add(tf.keras.layers.Dense(hidden_dims,kernel_initializer=hidden_initializer,kernel_regularizer='l1_l2'))
            disc.add(tf.keras.layers.LeakyReLU())
        disc.add(tf.keras.layers.Dense(1,kernel_initializer=output_initializer,activation=output_activation))
        return disc
        
    def build_pred(
        self,
        input_dims,
        celltype_dims,
        hidden_dims,
        pred_hidden_layers=1,
        hidden_initializer=tf.keras.initializers.HeNormal(),
        output_initializer=tf.keras.initializers.GlorotUniform(),
        output_activation='softmax'
    ):
        pred = tf.keras.Sequential()
        pred.add(tf.keras.Input(shape=input_dims,))
        for i in range(pred_hidden_layers):
            pred.add(tf.keras.layers.Dense(hidden_dims,kernel_initializer=hidden_initializer,kernel_regularizer='l1_l2'))
            pred.add(tf.keras.layers.LeakyReLU())
        pred.add(tf.keras.layers.Dense(celltype_dims,kernel_initializer=output_initializer,activation=output_activation))
        return pred

    def call(self, x):
        encoded = self.encoder(x)
        pred = self.pred(encoded)
        return encoded, pred

def kl_divergence(y_true, y_pred, axis=1):
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), np.inf)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.math.divide_no_nan(y_true, tf.reduce_sum(y_true, axis=axis, keepdims=True))
    y_pred = tf.math.divide_no_nan(y_pred, tf.reduce_sum(y_pred, axis=axis, keepdims=True))
    y_true = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1)
    return tf.reduce_sum(tf.multiply(y_true, tf.math.log(tf.math.divide_no_nan(y_true,y_pred))), axis=axis)
    
class SpointModel():
    def __init__(
        self, 
        st_ad, 
        sm_ad, 
        clusters, 
        used_genes, 
        spot_names, 
        st_batch_key=None,
        scvi_dims=64,
        latent_dims=32, 
        hidden_dims=512,
        ae_hidden_layers=3,
        disc_hidden_layers=1,
        pred_hidden_layers=1,
        sm_lr=3e-4,
        st_lr=3e-4,
        disc_lr=3e-4,
        always_batch_norm=False,
        rec_loss_axis=0
    ):
        self.st_ad = st_ad
        self.sm_ad = sm_ad
        self.scvi_dims=64
        self.spot_names = spot_names
        self.used_genes = used_genes
        self.clusters = clusters
        self.st_batch_key = st_batch_key
        self.mse_loss_func = None
        self.kl_loss_func = None
        self.cosine_infer_loss_func = None
        self.cosine_rec_loss_func = None
        self.sm_optimizer = None
        self.st_optimizer = None
        self.sm_train_rec_loss = None
        self.sm_train_infer_loss = None
        self.sm_train_loss = None
        self.sm_test_rec_loss = None
        self.sm_test_infer_loss = None
        self.sm_test_loss = None
        self.st_train_rec_loss = None
        self.st_test_rec_loss = None
        self.rec_loss_axis = rec_loss_axis
        self.model = self.build_model(scvi_dims,len(clusters),latent_dims,hidden_dims,sm_lr,st_lr,disc_lr,ae_hidden_layers,disc_hidden_layers,pred_hidden_layers,always_batch_norm)
        self.best_path = None
        self.history = pd.DataFrame(columns = ['sm_train_rec_loss','sm_train_infer_loss','sm_test_rec_loss','sm_test_infer_loss','sm_test_loss','st_train_rec_loss','is_best'])
        # logging.getLogger('spoint').setLevel(print)
        
        
    def get_scvi_latent(
        self,
        n_layers,
        n_latent,
        gene_likelihood,
        dispersion,
        max_epochs,
        early_stopping,
        batch_size,
    ):
        if self.st_batch_key is not None:
            if 'simulated' in self.st_ad.obs[self.st_batch_key]:
                raise ValueError(f'obs[{self.st_batch_key}] cannot include "real".')
            self.st_ad.obs["batch"] = self.st_ad.obs[self.st_batch_key].astype(str)
            self.sm_ad.obs["batch"] = 'simulated'
        else:
            self.st_ad.obs["batch"] = 'real'
            self.sm_ad.obs["batch"] = 'simulated'

        adata = sc.concat([self.st_ad,self.sm_ad])
        adata.layers["counts"] = adata.X.copy()

        scvi.model.SCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch"
        )

        vae = scvi.model.SCVI(adata, n_layers=n_layers, n_latent=n_latent, gene_likelihood=gene_likelihood,dispersion=dispersion)
        vae.train(max_epochs=max_epochs,early_stopping=early_stopping,batch_size=batch_size)
        adata.obsm["X_scVI"] = vae.get_latent_representation()

        st_scvi_ad = anndata.AnnData(adata[adata.obs['batch'] != 'simulated'].obsm["X_scVI"])
        sm_scvi_ad = anndata.AnnData(adata[adata.obs['batch'] == 'simulated'].obsm["X_scVI"])

        st_scvi_ad.obs = self.st_ad.obs
        st_scvi_ad.obsm = self.st_ad.obsm

        sm_scvi_ad.obs = self.sm_ad.obs
        sm_scvi_ad.obsm = self.sm_ad.obsm
        
        sm_scvi_ad = data_utils.check_data_type(sm_scvi_ad)
        st_scvi_ad = data_utils.check_data_type(st_scvi_ad)

        self.sm_data = sm_scvi_ad.X
        self.sm_labels = sm_scvi_ad.obsm['label'].values
        self.st_data = st_scvi_ad.X
        
        return sm_scvi_ad,st_scvi_ad

        
    def build_model(
        self,
        input_dims,
        celltype_dims,
        latent_dims,
        hidden_dims,
        sm_lr,
        st_lr,
        disc_lr,
        ae_hidden_layers,
        disc_hidden_layers,
        pred_hidden_layers,
        always_batch_norm
    ):
        self.mse_loss_func = tf.keras.losses.MeanSquaredError()
        self.kl_infer_loss_func = partial(kl_divergence, axis=1)
        self.kl_rec_loss_func = partial(kl_divergence, axis=self.rec_loss_axis)
        self.cosine_infer_loss_func = tf.keras.losses.CosineSimilarity(axis=1)
        self.cosine_rec_loss_func = tf.keras.losses.CosineSimilarity(axis=self.rec_loss_axis)
        self.bce_loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        self.sm_optimizer = tf.keras.optimizers.Adam(lr=sm_lr)
        self.st_optimizer = tf.keras.optimizers.Adam(lr=st_lr)
        self.disc_optimizer = tf.keras.optimizers.Adam(lr=disc_lr)

        self.sm_train_rec_loss = tf.keras.metrics.Mean(name='sm_train_rec_loss')
        self.sm_train_infer_loss = tf.keras.metrics.Mean(name='sm_train_infer_loss')
        self.sm_train_loss = tf.keras.metrics.Mean(name='sm_train_loss')
        self.sm_test_rec_loss = tf.keras.metrics.Mean(name='sm_test_rec_loss')
        self.sm_test_infer_loss = tf.keras.metrics.Mean(name='sm_test_infer_loss')
        self.sm_test_loss = tf.keras.metrics.Mean(name='sm_test_loss')
        self.st_train_rec_loss = tf.keras.metrics.Mean(name='st_train_rec_loss')
        self.st_test_rec_loss = tf.keras.metrics.Mean(name='st_test_rec_loss')
        self.disc_train_loss = tf.keras.metrics.Mean(name='disc_train_loss')
        self.disc_test_loss = tf.keras.metrics.Mean(name='disc_test_loss')
        return BaseModel(input_dims,latent_dims,hidden_dims,celltype_dims,ae_hidden_layers,disc_hidden_layers,pred_hidden_layers,always_batch_norm)
    
    def build_dataset(self,st_data,sm_data,sm_labels):
        x_train,y_train,x_test,y_test = data_utils.split_shuffle_data(np.array(sm_data,dtype=np.float32),np.array(sm_labels,dtype=np.float32))
        train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        st_ds = tf.data.Dataset.from_tensor_slices(np.array(st_data, dtype=np.float32))
        return train_ds, test_ds, st_ds
    
    @tf.function
    def train_sm(self, data, labels, d_loss, rec_w=1, infer_w=1, d_w=1):
        with tf.GradientTape() as tape:
            latent = self.model.encoder(data, training=True)
            rec_data = self.model.decoder(latent, training=True)
            predictions = self.model.pred(latent, training=True)
            rec_loss = tf.reduce_mean(self.kl_rec_loss_func(data, rec_data)) + tf.reduce_mean(self.cosine_rec_loss_func(data,rec_data))
            infer_loss = tf.reduce_mean(self.kl_infer_loss_func(labels,predictions)) + tf.reduce_mean(self.cosine_infer_loss_func(labels,predictions))
            loss = rec_w*rec_loss + infer_w*infer_loss + d_w*d_loss
        gradients = tape.gradient(loss, self.model.encoder.trainable_variables+self.model.decoder.trainable_variables+self.model.pred.trainable_variables)
        self.sm_optimizer.apply_gradients(zip(gradients, self.model.encoder.trainable_variables+self.model.decoder.trainable_variables+self.model.pred.trainable_variables))
        self.sm_train_rec_loss(rec_loss)
        self.sm_train_infer_loss(infer_loss)
        self.sm_train_loss(loss)
        return latent

    @tf.function
    def train_st(self,data, d_loss,rec_w=1,d_w=1):
        with tf.GradientTape() as tape:
            latent = self.model.encoder(data, training=True)
            rec_data = self.model.decoder(latent, training=True)
            rec_loss = tf.reduce_mean(self.kl_rec_loss_func(data,rec_data)) + tf.reduce_mean(self.cosine_rec_loss_func(data,rec_data))
            loss = tf.add(rec_w*rec_loss,d_w*d_loss)
        gradients = tape.gradient(loss, self.model.encoder.trainable_variables+self.model.decoder.trainable_variables)
        self.st_optimizer.apply_gradients(zip(gradients, self.model.encoder.trainable_variables+self.model.decoder.trainable_variables))
        self.st_train_rec_loss(rec_loss)
        return latent
    
    @tf.function
    def train_disc(self,latent_1, latent_2):
        with tf.GradientTape() as tape:
            latent = tf.concat((latent_1,latent_2),axis=0)
            labels = tf.concat((tf.ones(latent_1.shape[0]),tf.zeros(latent_2.shape[0])),axis=0)
            pred_d = self.model.disc(latent, training=True)
#             pred_d = tf.reshape(pred_d,[latent.shape[0]])
            disc_loss = self.bce_loss_func(labels,pred_d)
        gradients = tape.gradient(disc_loss, self.model.disc.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients, self.model.disc.trainable_variables))
        self.disc_train_loss(disc_loss)
        return disc_loss
        
    @tf.function
    def test_sm(self,data, labels, rec_w=1, infer_w=1):
        latent = self.model.encoder(data, training=False)
        rec_data = self.model.decoder(latent, training=False)
        predictions = self.model.pred(latent, training=False)
        rec_loss = rec_w*tf.reduce_mean(self.kl_rec_loss_func(data,rec_data)) + tf.reduce_mean(self.cosine_rec_loss_func(data,rec_data))
        infer_loss = infer_w*tf.reduce_mean(self.kl_infer_loss_func(labels,predictions)) + tf.reduce_mean(self.cosine_infer_loss_func(labels,predictions))
        loss = rec_loss + infer_loss
        self.sm_test_rec_loss(rec_loss)
        self.sm_test_infer_loss(infer_loss)
        self.sm_test_loss(loss)
        return latent
        
    @tf.function
    def test_st(self,data, rec_w=1):
        latent = self.model.encoder(data, training=False)
        rec_data = self.model.decoder(latent, training=False)
        rec_loss = rec_w*tf.reduce_mean(self.kl_rec_loss_func(data,rec_data)) + tf.reduce_mean(self.cosine_rec_loss_func(data,rec_data))
        self.st_test_rec_loss(rec_loss)
        return latent
    
    def test_disc(self,latent_1, latent_2):
        latent = tf.concat((latent_1,latent_2),axis=0)
        labels = tf.concat((tf.ones(latent_1.shape[0]),tf.zeros(latent_2.shape[0])),axis=0)
        pred_d = self.model.disc(latent, training=False)
        disc_loss = self.bce_loss_func(labels,pred_d)
        self.disc_test_loss(disc_loss)
        return disc_loss

    @staticmethod
    def add_noise(proportion,minval=-0.1,maxval=0.1):
        return tf.clip_by_value(proportion + tf.random.uniform(shape=proportion.shape,minval=minval,maxval=maxval),0,1)
    
    def train_model_by_step(
        self,
        max_steps=50000,
        save_mode='best',
        save_path=None,
        prefix=None,
        sm_step=4,
        st_step=1,
        disc_step=1,
        test_step_gap=10,
        convergence=0.001,
        early_stop=True,
        early_stop_max=20,
        sm_lr=None,
        st_lr=None,
        disc_lr=None,
        batch_size=4096,
        rec_w=0.5, 
        infer_w=1,
        d_w=1,
        m_w=1000,
        noise=None
    ):
        st_ds_size = self.st_ds.cardinality().numpy()
        train_ds_size = self.train_ds.cardinality().numpy()
        if len(self.history) > 0:
            best_ind = np.where(self.history['is_best'] == 'True')[0][-1]
            best_loss = self.history['sm_test_infer_loss'][best_ind]
            best_rec_loss = self.history['st_test_rec_loss'][best_ind]
        else:
            best_loss = np.inf
            best_rec_loss = np.inf
        early_stop_count = 0
        if save_path is None:
            save_path = 'Spoint_models'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if sm_lr is not None:
            self.sm_optimizer.lr.assign(sm_lr)
        if st_lr is not None:
            self.st_optimizer.lr.assign(st_lr)
        if disc_lr is not None:
            self.disc_optimizer.lr.assign(disc_lr)
        
        sm_shuffle_step = max(int(train_ds_size/(batch_size*sm_step)),1)
        st_shuffle_step = max(int(st_ds_size/(batch_size*st_step)),1)
        
        d_loss = 0
        
        for step in range(max_steps):
            if step % sm_shuffle_step == 0:
                train_ds_iter = itertools.cycle(self.train_ds.shuffle(train_ds_size).batch(batch_size))
            if step % st_shuffle_step == 0:
                st_ds_iter = itertools.cycle(self.st_ds.shuffle(st_ds_size).batch(batch_size))
            self.sm_train_rec_loss.reset_states()
            self.sm_train_infer_loss.reset_states()
            self.sm_train_loss.reset_states()
            self.sm_test_rec_loss.reset_states()
            self.sm_test_infer_loss.reset_states()
            self.sm_test_loss.reset_states()
            self.st_train_rec_loss.reset_states()
            self.st_test_rec_loss.reset_states()
            self.disc_train_loss.reset_states()
            self.disc_test_loss.reset_states()
                
            if step % test_step_gap == 0:
                for exp in self.st_ds.batch(batch_size):
                    st_latent = self.test_st(exp, rec_w=rec_w)
                for exp, proportion in self.test_ds.batch(batch_size):
                    sm_latent = self.test_sm(exp,proportion, rec_w=rec_w, infer_w=infer_w)
                mmd_loss = compute_mmd(st_latent,sm_latent)*m_w
                # for i in range(disc_step):
                #     d_loss = self.test_disc(st_latent, sm_latent)
                    
                current_loss = self.sm_test_infer_loss.result()
                current_rec_loss = self.st_test_rec_loss.result()

                best_flag='False'
                # if (best_loss - current_loss > convergence) & (best_rec_loss - current_rec_loss > convergence):
                if best_loss - current_loss > convergence:
                    if best_loss > current_loss:
                        best_loss = current_loss
                    # if best_rec_loss > current_rec_loss:
                    #     best_rec_loss = current_rec_loss
                    best_flag='True'
                    print('### Update best model')
                    early_stop_count = 0
                    old_best_path = self.best_path
                    if prefix is not None:
                        self.best_path = os.path.join(save_path,prefix+'_'+f'celleagle_weights_step{step}.h5')
                    else:
                        self.best_path = os.path.join(save_path,f'celleagle_weights_step{step}.h5')
                    if save_mode == 'best':
                        if old_best_path is not None:
                            if os.path.exists(old_best_path):
                                os.remove(old_best_path)
                        self.model.save_weights(self.best_path)
                else:
                    early_stop_count += 1
                if save_mode == 'all':
                    if prefix != '':
                        self.best_path = os.path.join(save_path,prefix+'_'+f'celleagle_weights_step{step}.h5')
                    else:
                        self.model.save_weights(os.path.join(save_path,f'celleagle_weights_step{step}.h5'))
                self.history = self.history.append({
                    'sm_train_rec_loss':self.sm_train_rec_loss.result(),
                    'sm_train_infer_loss':self.sm_train_infer_loss.result(),
                    'sm_train_loss':self.sm_train_loss.result(),
                    'sm_test_rec_loss':self.sm_test_rec_loss.result(),
                    'sm_test_infer_loss':self.sm_test_infer_loss.result(),
                    'sm_test_loss':self.sm_test_loss.result(),
                    'st_train_rec_loss':self.st_train_rec_loss.result(),
                    'st_test_rec_loss':self.st_test_rec_loss.result(),
                    'disc_train_rec_loss':self.disc_train_loss.result(),
                    'disc_test_rec_loss':self.disc_test_loss.result(),
                    'is_best':best_flag
                }, ignore_index=True)
                # print(
                #     f'Step {step + 1}: sm_infer_loss:{self.sm_test_infer_loss.result():.3f}, sm_rec_loss: {self.sm_test_rec_loss.result():.3f}, st_rec_loss: {self.st_test_rec_loss.result():.3f}, disc_loss: {self.disc_test_loss.result():.3f}, mmd_loss: {mmd_loss:.3f}'
                # )
                print(
                    f'Step {step + 1} - loss: {self.sm_test_infer_loss.result():.3f}'
                )
                if (early_stop_count > early_stop_max) and early_stop:
                    print('Stop trainning because of loss convergence')
                    break
            
            for i in range(st_step):
                exp = next(st_ds_iter)
                st_latent = self.train_st(exp, mmd_loss, rec_w=rec_w, d_w=d_w)
            for i in range(sm_step):
                exp, proportion = next(train_ds_iter)
                sm_latent = self.train_sm(exp, proportion, mmd_loss, rec_w=rec_w, infer_w=infer_w, d_w=d_w)
            mmd_loss = compute_mmd(st_latent,sm_latent)*m_w
            # for i in range(disc_step):
            #     d_loss = self.train_disc(st_latent, sm_latent)
            
    def train(
        self,
        max_steps=50000,
        save_mode='best',
        save_path=None,
        prefix=None,
        sm_step=4,
        st_step=1,
        disc_step=1,
        test_step_gap=10,
        convergence=0.001,
        early_stop=True,
        early_stop_max=20,
        sm_lr=None,
        st_lr=None,
        disc_lr=None,
        batch_size=4096,
        rec_w=0.5, 
        infer_w=1,
        d_w=1,
        m_w=1000,
        noise=None,
        scvi_layers=2,
        scvi_latent=64,
        scvi_gene_likelihood='zinb',
        scvi_dispersion='gene-batch',
        scvi_max_epochs=100,
        scvi_early_stopping=True,
        scvi_batch_size=4096,
    ):
        
        self.get_scvi_latent(
            n_layers=scvi_layers,
            n_latent=scvi_latent,
            gene_likelihood=scvi_gene_likelihood,
            dispersion=scvi_dispersion,
            max_epochs=scvi_max_epochs,
            early_stopping=scvi_early_stopping,
            batch_size=scvi_batch_size,
        )
        self.train_ds, self.test_ds, self.st_ds = self.build_dataset(self.st_data,self.sm_data,self.sm_labels)
        self.train_model_by_step(
            max_steps=max_steps,
            save_mode=save_mode,
            save_path=save_path,
            prefix=prefix,
            sm_step=sm_step,
            st_step=st_step,
            disc_step=disc_step,
            test_step_gap=test_step_gap,
            convergence=convergence,
            early_stop=early_stop,
            early_stop_max=early_stop_max,
            sm_lr=sm_lr,
            st_lr=st_lr,
            disc_lr=disc_lr,
            batch_size=batch_size,
            rec_w=rec_w, 
            infer_w=infer_w,
            d_w=d_w,
            m_w=m_w
        )
    
    def eval_model(self,model_path=None,use_best_model=True,batch_size=4096,metric='pcc'):
        if metric=='pcc':
            metric_name = 'PCC'
            func = metrics.pcc
        if metric=='spcc':
            metric_name = 'SPCC'
            func = metrics.spcc
        if metric=='mae':
            metric_name = 'MAE'
            func = metrics.mae
        if metric=='js':
            metric_name = 'JS'
            func = metrics.js
        if metric=='rmse':
            metric_name = 'RMSE'
            func = metrics.rmse
        if metric=='ssim':
            metric_name = 'SSIM'
            func = metrics.ssim
        
        # call model first before load weights
        self.model(self.st_data, training=False)
        
        if model_path is not None:
            self.model.load_weights(model_path)
        elif use_best_model:
            self.model.load_weights(self.best_path)
        pre = []
        prop = []
        for exp_batch, prop_batch in self.test_ds.batch(batch_size):
            latent_tmp = self.model.encoder(exp_batch, training=False)
            pre_tmp = self.model.pred(latent_tmp, training=False).numpy()
            pre.extend(pre_tmp)
            prop.extend(prop_batch.numpy())
        pre = np.array(pre)
        prop = np.array(prop)
        metric_list = []
        for i,c in enumerate(self.clusters):
            metric_list.append(func(pre[:,i],prop[:,i]))
        print('### Evaluate model with simulation data')
        for i in range(len(metric_list)):
            print(f'{metric_name} of {self.clusters[i]}, {metric_list[i]}')
            
    def plot_training_history(self,save=None,return_fig=False,show=True,dpi=300):
        if len(self.history) > 0:
            fig, ax = plt.subplots()
            plt.plot(np.arange(len(self.history)), self.history['sm_test_infer_loss'], label='sm_test_infer_loss')
            plt.plot(np.arange(len(self.history)), self.history['st_test_rec_loss'], label='st_test_rec_loss')
            plt.xlabel('Epochs')
            plt.ylabel('Losses')
            plt.title('Training history')
            plt.legend()
            if save is not None:
                plt.savefig(save,bbox_inches='tight',dpi=dpi)
            if show:
                plt.show()
            plt.close()
            if return_fig:
                return fig
        else:
            print('History is empty, training model first')
            

    def deconv_spatial(self,st_data=None,model_path=None,use_best_model=True,add_obs=True,add_uns=True):
        if st_data is None:
            st_data = self.st_data
        # st_data_norm = data_utils.normalize_mtx(st_data,target_sum=1e4)
        self.model(st_data, training=False)
        if model_path is not None:
            self.model.load_weights(model_path)
        elif use_best_model:
            self.model.load_weights(self.best_path)
        latent = self.model.encoder(st_data, training=False)
        pre = self.model.pred(latent, training=False).numpy()
        pre = pd.DataFrame(pre,columns=self.clusters,index=self.st_ad.obs_names)
        self.st_ad.obs[pre.columns] = pre.values
        self.st_ad.uns['celltypes'] = list(pre.columns)
        return pre
