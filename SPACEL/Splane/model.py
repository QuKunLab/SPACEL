import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os 
import matplotlib
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import seaborn as sns
import squidpy as sq
from .graph import get_GNN_inputs,get_GNN_kernel,kl_divergence,split_train_test_data,GraphConvolutionModel,DiscriminationModel
from .utils import generate_celltype_ad_list,cal_celltype_weight,clustering

class SplaneModel():
    def __init__(
        self,
        expr_ad_list,
        n_clusters,
        X,
        G,
        graph,
        support,
        slice_class_onehot,
        nb_mask,
        train_mask,
        test_mask,
        celltype_weights,
        morans_mean,
        lr,
        latent_dim,
        hidden_dims,
        gnn_dropout
    ):
        self.expr_ad_list = expr_ad_list
        self.model_g = GraphConvolutionModel(X.shape[1],G,support,latent_dims=latent_dim,hidden_dims=hidden_dims,dropout=gnn_dropout)
        self.model_d = DiscriminationModel(slice_class_onehot,latent_dims=latent_dim,hidden_dims=hidden_dims)
        self.graph = graph
        self.slice_class_onehot = slice_class_onehot
        self.nb_mask = nb_mask
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.celltype_weights = celltype_weights
        self.morans_mean = morans_mean
        self.best_path = None
        self.cos_loss_obj = tf.keras.losses.CosineSimilarity(axis=0,reduction=tf.keras.losses.Reduction.NONE)
        self.d_loss_obj = tf.keras.losses.CategoricalCrossentropy()
        self.n_clusters = n_clusters
        self.Cluster = KMeans(n_clusters=self.n_clusters,n_init=10,tol=1e-3,algorithm='full',max_iter=1000,random_state=42)
        self.optimizer_g=tf.keras.optimizers.RMSprop(learning_rate=lr)
        self.optimizer_d=tf.keras.optimizers.RMSprop(learning_rate=lr)
        
    # @tf.function
    def train_model_g(self,d_l,simi_l):
        with tf.GradientTape(persistent=False) as tape:
            encoded, decoded = self.model_g(self.graph,training=True,decode_only=False)
            # print(encoded)
            y_disc = self.model_d(encoded,training=True)
            # print(y_disc)
            d_loss_value = self.d_loss_obj(y_true=self.slice_class_onehot, y_pred=y_disc)
            decoded_mask = tf.gather_nd(decoded, tf.where(self.train_mask))
            x_mask = tf.gather_nd(self.graph[0], tf.where(self.train_mask))
            f_adj = tf.matmul(encoded, tf.transpose(encoded))
            simi_loss = -tf.reduce_mean(tf.gather_nd(f_adj,self.nb_mask.T)) + tf.reduce_mean(tf.abs(tf.math.subtract(tf.gather(encoded,self.nb_mask[0]),tf.gather(encoded,self.nb_mask[1]))),axis=0)
            g_loss_value = tf.reduce_sum(self.celltype_weights*self.cos_loss_obj(y_true=x_mask, y_pred=decoded_mask))+tf.reduce_sum(self.celltype_weights*kl_divergence(y_true=x_mask, y_pred=decoded_mask, axis=0)) + simi_l*simi_loss
            total_loss = g_loss_value - d_l*d_loss_value
            # print(simi_loss,d_loss_value,total_loss)
        gradients_g = tape.gradient(total_loss, self.model_g.trainable_variables)
        self.optimizer_g.apply_gradients(zip(gradients_g, self.model_g.trainable_variables))
        return total_loss
    
    # @tf.function
    def train_model_d(self):
        with tf.GradientTape(persistent=False) as tape:
            encoded = self.model_g(self.graph,training=True,encode_only=True)
            y_disc = self.model_d(encoded,training=True)
            d_loss_value = self.d_loss_obj(y_true=self.slice_class_onehot, y_pred=y_disc)
        gradients_d = tape.gradient(d_loss_value, self.model_d.trainable_variables)
        self.optimizer_d.apply_gradients(zip(gradients_d, self.model_d.trainable_variables))
        return d_loss_value
    
    # @tf.function
    def test_model(self,d_l,simi_l):
        encoded, decoded = self.model_g(self.graph,training=False,decode_only=False)
        y_disc = self.model_d(encoded,training=False)
        d_loss_value = self.d_loss_obj(y_true=self.slice_class_onehot, y_pred=y_disc)
        decoded_mask = tf.gather_nd(decoded, tf.where(self.test_mask))
        x_mask = tf.gather_nd(self.graph[0], tf.where(self.test_mask))
        ll = tf.math.equal(tf.math.argmax(self.slice_class_onehot, -1), tf.math.argmax(y_disc, -1))
        accuarcy = tf.reduce_mean(tf.cast(ll, dtype=tf.float32))
        f_adj = tf.matmul(encoded, tf.transpose(encoded))
        simi_loss = -tf.reduce_mean(tf.gather_nd(f_adj,self.nb_mask.T)) + tf.reduce_mean(tf.abs(tf.math.subtract(tf.gather(encoded,self.nb_mask[0]),tf.gather(encoded,self.nb_mask[1]))))
        db_loss = clustering(self.Cluster, encoded.numpy())
        g_loss_value = tf.reduce_sum(self.celltype_weights*self.cos_loss_obj(y_true=x_mask, y_pred=decoded_mask))+tf.reduce_sum(self.celltype_weights*kl_divergence(y_true=x_mask, y_pred=decoded_mask, axis=0)) + simi_l*simi_loss
        total_loss = g_loss_value - d_l*d_loss_value
    
        return total_loss, g_loss_value, d_loss_value, accuarcy, simi_loss, db_loss, encoded, decoded

    def train(
        self,
        max_epochs=1000,
        convergence=0.0001,
        db_convergence=0,
        early_stop_epochs=10,
        d_l=0.2,
        simi_l=None,
        g_step = 1,
        d_step = 1,
        plot_step=5,
        save_path = 'Splane_models',
        prefix=None
    ):
        best_loss = np.inf
        best_db_loss = np.inf
        best_simi_loss = np.inf
        simi_l = 1/np.mean(self.morans_mean)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        early_stop_count = 0
        for epoch in range(max_epochs):

            for _ in range(g_step):
                train_total_loss = self.train_model_g(d_l=d_l, simi_l=simi_l)
            for _ in range(d_step):
                self.train_model_d()

            if epoch % plot_step == 0:
                test_total_loss, test_loss_g_value, test_loss_d_value, test_acc, simi_loss, db_loss, encoded, decoded = self.test_model(d_l=d_l, simi_l=simi_l)
                current_loss = test_loss_g_value
                current_db_loss = db_loss
                if (best_loss - current_loss > convergence) & (best_db_loss - current_db_loss > db_convergence):
                    if best_loss > current_loss:
                        best_loss = current_loss
                    if best_db_loss > current_db_loss:
                        best_db_loss = current_db_loss
                    print('### Update best model')
                    old_best_path = self.best_path
                    early_stop_count = 0
                    if prefix is not None:
                        self.best_path = os.path.join(save_path,prefix+'_'+f'Splane_weights_epoch{epoch}.h5')
                    else:
                        self.best_path = os.path.join(save_path,f'Splane_weights_epoch{epoch}.h5')
                    if old_best_path is not None:
                        if os.path.exists(old_best_path):
                            os.remove(old_best_path)
                    self.model_g.save_weights(self.best_path)
                else:
                    early_stop_count += 1

                print("Epoch {} train g loss={} g loss={} d loss={} acc={} simi loss={} db loss={}".format(epoch, test_total_loss, test_loss_g_value, test_loss_d_value, test_acc, simi_loss, db_loss))
                if early_stop_count > early_stop_epochs:
                    print('Stop trainning because of loss convergence')
                    break
                    
    def identify_spatial_domain(self,colors=None,key=None):
        if colors is None:
            if self.n_clusters > 10:
                colors = [matplotlib.colors.to_hex(c) for c in sns.color_palette('tab20',n_colors=self.n_clusters)]
            else:
                colors = [matplotlib.colors.to_hex(c) for c in sns.color_palette('tab10',n_colors=self.n_clusters)]
            color_map = pd.DataFrame(colors,index=np.arange(self.n_clusters),columns=['color'])
        if key is None:
            key = 'spatial_domain'
        self.model_g(self.graph)
        self.model_g.load_weights(self.best_path)
        encoded, decoded = self.model_g(self.graph,training=False,decode_only=False,encode_only=False)
        clusters = self.Cluster.fit_predict(encoded.numpy())
        loc_index = 0
        for i in range(len(self.expr_ad_list)):
            if key in self.expr_ad_list[i].obs.columns:
                self.expr_ad_list[i].obs = self.expr_ad_list[i].obs.drop(columns=key)
            self.expr_ad_list[i].obs[key] = clusters[loc_index:loc_index+self.expr_ad_list[i].shape[0]]
            self.expr_ad_list[i].obs[key] = pd.Categorical(self.expr_ad_list[i].obs[key])
            self.expr_ad_list[i].uns[f'{key}_colors'] = [color_map.loc[c,'color'] for c in self.expr_ad_list[i].obs['spatial_domain'].cat.categories]
            loc_index += self.expr_ad_list[i].shape[0]
            
        
def init_model(
    expr_ad_list:list,
    n_clusters:int,
    k:int=2,
    use_weight=True,
    train_prop:float=0.5,
    n_neighbors=6,
    min_prop=0.01,
    lr:float=3e-3,
    latent_dim:int=16,
    hidden_dims:int=64,
    gnn_dropout:float=0.8,
    seed=42
)->SplaneModel:
    
    print('Setting global seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    for expr_ad in expr_ad_list:
        if 'spatial_connectivities' not in expr_ad.obsp.keys():
            sq.gr.spatial_neighbors(expr_ad,coord_type='grid',n_neighs=n_neighbors)
    celltype_ad_list = generate_celltype_ad_list(expr_ad_list,min_prop)
    celltype_weights,morans_mean = cal_celltype_weight(celltype_ad_list)
    if not use_weight:
        celltype_weights = np.ones(len(celltype_weights))/len(celltype_weights)
    X,A,nb_mask,slice_class_onehot = get_GNN_inputs(celltype_ad_list)
    X_filtered, graph, G, support = get_GNN_kernel(X,A,k=k)
    train_mask,test_mask = split_train_test_data(X,train_prop=0.5)
    return SplaneModel(
        expr_ad_list,
        n_clusters,
        X_filtered,
        G,
        graph,
        support,
        slice_class_onehot,
        nb_mask,
        train_mask,
        test_mask,
        celltype_weights,
        morans_mean,
        lr,
        latent_dim,
        hidden_dims,
        gnn_dropout
    )
