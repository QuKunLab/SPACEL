import numpy as np
import pandas as pd
import random
import os 
import matplotlib
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import seaborn as sns
import squidpy as sq
from .utils import clustering
import math
import time
import numpy as np
from tqdm import tqdm
from time import strftime, localtime
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, support, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.support = support
        self.weight = Parameter(torch.FloatTensor(in_features * support, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, basis):
        supports = list()
        for i in range(self.support):
            supports.append(basis[i].matmul(features))
        supports = torch.cat(supports, dim=1)
        output = torch.spmm(supports, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
            
class Splane_GCN(nn.Module):
    def __init__(self,feature_dims,support,latent_dims=8,hidden_dims=64,dropout=0.8):
        super(Splane_GCN, self).__init__()
        self.feature_dims = feature_dims
        self.support = support
        self.latent_dims=latent_dims
        self.hidden_dims=hidden_dims
        self.dropout = dropout
        self.encode_gc1 = GraphConvolution(feature_dims, hidden_dims, support)
        self.encode_gc2 = GraphConvolution(hidden_dims, latent_dims, support)
        self.decode_gc1 = GraphConvolution(latent_dims, hidden_dims, support)
        self.decode_gc2 = GraphConvolution(hidden_dims, feature_dims, support)
        
        nn.init.kaiming_normal_(self.encode_gc1.weight)
        nn.init.xavier_uniform_(self.encode_gc2.weight)
        nn.init.kaiming_normal_(self.decode_gc1.weight)
        nn.init.xavier_uniform_(self.decode_gc2.weight)
        
    @staticmethod
    def l2_activate(x,dim):
        
        def scale(z):
            zmax = z.max(1, keepdims=True).values
            zmin = z.min(1, keepdims=True).values
            z_std = torch.nan_to_num(torch.div(z - zmin,(zmax - zmin)),0)
            return z_std
        
        x = scale(x)
        x = F.normalize(x, p=2, dim=1)
        return x
        
    def encode(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.encode_gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.encode_gc2(x, adj)
        return self.l2_activate(x, dim=1)
    
    def decode(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.decode_gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.decode_gc2(x, adj)
        return x

    def forward(self, x, adj):
        z = self.encode(x, adj)
        x_ = self.decode(z, adj)
        return z, x_
    
class Splane_Disc(nn.Module):
    def __init__(self,label,latent_dims=8,hidden_dims=64,dropout=0.5):
        super(Splane_Disc, self).__init__()
        self.latent_dims=latent_dims
        self.hidden_dims=hidden_dims
        self.dropout = dropout
        self.class_num = label.shape[1]
        self.disc = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dims),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dims),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, self.class_num)
        )

    def forward(self, x):
        x = self.disc(x)
        y = F.softmax(x, dim=1)
        return y

class SplaneModel():
    def __init__(
        self,
        expr_ad_list,
        n_clusters,
        X,
        graph,
        support,
        slice_class_onehot,
        nb_mask,
        train_idx,
        test_idx,
        celltype_weights,
        morans_mean,
        lr,
        l1,
        l2,
        latent_dim,
        hidden_dims,
        gnn_dropout,
        use_gpu
    ):
        self.expr_ad_list = expr_ad_list
        self.model_g = Splane_GCN(X.shape[1],support,latent_dims=latent_dim,hidden_dims=hidden_dims,dropout=gnn_dropout)
        self.model_d = Splane_Disc(slice_class_onehot,latent_dims=latent_dim,hidden_dims=hidden_dims)
        self.graph = graph
        self.slice_class_onehot = slice_class_onehot
        self.nb_mask = nb_mask
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.celltype_weights = torch.tensor(celltype_weights)
        self.morans_mean = morans_mean
        self.best_path = None
        self.cos_loss_obj = F.cosine_similarity
        self.d_loss_obj = F.cross_entropy
        self.n_clusters = n_clusters
        self.Cluster = KMeans(n_clusters=self.n_clusters,n_init=10,tol=1e-3,algorithm='full',max_iter=1000,random_state=42)
        self.optimizer_g = optim.RMSprop(self.model_g.parameters(),lr=lr)
        self.optimizer_d = optim.RMSprop(self.model_d.parameters(),lr=lr)
        self.l1 = l1
        self.l2 = l2
        if use_gpu:
            self.model_g = self.model_g.cuda()
            self.model_d = self.model_d.cuda()
            self.celltype_weights = self.celltype_weights.cuda()
    
    @staticmethod
    def kl_divergence(y_true, y_pred, dim=0):
        y_pred = torch.clip(y_pred, torch.finfo(torch.float32).eps)
        y_true = y_true.to(y_pred.dtype)
        y_true = torch.nan_to_num(torch.div(y_true, y_true.sum(dim, keepdims=True)),0)
        y_pred = torch.nan_to_num(torch.div(y_pred, y_pred.sum(dim, keepdims=True)),0)
        y_true = torch.clip(y_true, torch.finfo(torch.float32).eps, 1)
        y_pred = torch.clip(y_pred, torch.finfo(torch.float32).eps, 1)
        return torch.mul(y_true, torch.log(torch.nan_to_num(torch.div(y_true, y_pred)))).mean(dim)
        
    def train_model_g(self,d_l,simi_l):
        self.model_g.train()
        self.optimizer_g.zero_grad()
        encoded, decoded = self.model_g(self.graph[0],self.graph[1:])
        y_disc = self.model_d(encoded)
        d_loss = F.cross_entropy(self.slice_class_onehot, y_disc)
        decoded_mask = decoded[self.train_idx]
        x_mask = self.graph[0][self.train_idx]
        f_adj = torch.matmul(encoded, torch.transpose(encoded,0,1))
        simi_loss = -torch.mean(f_adj[self.nb_mask[0],self.nb_mask[1]]) + torch.mean(torch.abs(encoded[self.nb_mask[0]]-encoded[self.nb_mask[1]]))
        g_loss = torch.sum(self.celltype_weights*F.cosine_similarity(x_mask, decoded_mask,dim=0))+torch.sum(self.celltype_weights*self.kl_divergence(x_mask, decoded_mask, dim=0)) + simi_l*simi_loss
        
        regularization_params = torch.cat([
            torch.cat([x.view(-1) for x in self.model_g.encode_gc1.parameters()]),
            torch.cat([x.view(-1) for x in self.model_g.decode_gc1.parameters()])
        ])
        l1_regularization = self.l1 * torch.norm(regularization_params, 1)
        l2_regularization = self.l2 * torch.norm(regularization_params, 2)

        l1_l2_loss = l1_regularization + l2_regularization
        
        total_loss = g_loss - d_l*d_loss # + l1_l2_loss
        total_loss.backward()
        self.optimizer_g.step()
        return total_loss
    
    def train_model_d(self,):
        self.model_d.train()
        self.optimizer_d.zero_grad()
        encoded, decoded = self.model_g(self.graph[0],self.graph[1:])
        y_disc = self.model_d(encoded)
        d_loss = F.cross_entropy(self.slice_class_onehot, y_disc)
        d_loss.backward()
        self.optimizer_d.step()
        return d_loss
    
    def test_model(self,d_l,simi_l):
        self.model_g.eval()
        self.model_d.eval()
        encoded, decoded = self.model_g(self.graph[0],self.graph[1:])
        y_disc = self.model_d(encoded)
        d_loss = F.cross_entropy(self.slice_class_onehot, y_disc)
        decoded_mask = decoded[self.test_idx]
        x_mask = self.graph[0][self.test_idx]
        ll = torch.eq(torch.argmax(self.slice_class_onehot, -1), torch.argmax(y_disc, -1))
        accuarcy = ll.to(torch.float32).mean()
        f_adj = torch.matmul(encoded, torch.transpose(encoded,0,1))
        simi_loss = -torch.mean(f_adj[self.nb_mask[0],self.nb_mask[1]]) + torch.mean(torch.abs(encoded[self.nb_mask[0]]-encoded[self.nb_mask[1]]))
        g_loss = torch.sum(self.celltype_weights*F.cosine_similarity(x_mask, decoded_mask,dim=0))+torch.sum(self.celltype_weights*self.kl_divergence(x_mask, decoded_mask, dim=0)) + simi_l*simi_loss
        total_loss = g_loss - d_l*d_loss
        db_loss = clustering(self.Cluster, encoded.cpu().detach().numpy())
        return total_loss, g_loss, d_loss, accuarcy, simi_loss, db_loss, encoded, decoded
    
    def train(
        self,
        max_epochs=300,
        convergence=0.0001,
        db_convergence=0,
        early_stop_epochs=10,
        d_l=0.5,
        simi_l=None,
        g_step = 1,
        d_step = 1,
        plot_step=5,
        save_path=None,
        prefix=None
    ):

    """Training Splane model.

    Args:
        max_steps: The max step of training. The training process will be stop when achive max step.
        convergence: The total loss threshold for early stop.
        db_convergence: The DBS threshold for early stop.
        early_stop_epochs: The max epochs of loss difference less than convergence.
        d_l: The weight of discriminator loss.
        simi_l: The weight of similarity loss.
        plot_step: The interval steps of training.
        save_path: A string representing the path directory where the model is saved.
        prefix: A string added to the prefix of file name of saved model.
    Returns:
        ``None``
    """
        best_loss = np.inf
        best_db_loss = np.inf
        best_simi_loss = np.inf
        simi_l = 1/np.mean(self.morans_mean)
        
        if save_path is None:
            save_path = os.path.join(tempfile.gettempdir() ,'Splane_models_'+strftime("%Y%m%d%H%M%S",localtime()))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        early_stop_count = 0
        pbar = tqdm(range(max_epochs))
        for epoch in pbar:

            for _ in range(g_step):
                train_total_loss = self.train_model_g(d_l=d_l, simi_l=simi_l)
            for _ in range(d_step):
                self.train_model_d()

            if epoch % plot_step == 0:
                test_total_loss, test_g_loss, test_d_loss, test_acc, simi_loss, db_loss, encoded, decoded = self.test_model(d_l=d_l, simi_l=simi_l)
                current_loss = test_g_loss.cpu().detach().numpy()
                current_db_loss = db_loss
                if (best_loss - current_loss > convergence) & (best_db_loss - current_db_loss > db_convergence):
                    if best_loss > current_loss:
                        best_loss = current_loss
                    if best_db_loss > current_db_loss:
                        best_db_loss = current_db_loss
                    pbar.set_description("The best epoch {0} total loss={1:.3f} g loss={2:.3f} d loss={3:.3f} d acc={4:.3f} simi loss={5:.3f} db loss={6:.3f}".format(epoch, test_total_loss, test_g_loss, test_d_loss, test_acc, simi_loss, db_loss),refresh=True)
                    old_best_path = self.best_path
                    early_stop_count = 0
                    if prefix is not None:
                        self.best_path = os.path.join(save_path,prefix+'_'+f'Splane_weights_epoch{epoch}.h5')
                    else:
                        self.best_path = os.path.join(save_path,f'Splane_weights_epoch{epoch}.h5')
                    if old_best_path is not None:
                        if os.path.exists(old_best_path):
                            os.remove(old_best_path)
                    torch.save(self.model_g.state_dict(), self.best_path)
                else:
                    early_stop_count += 1
                

                # print("Epoch {} train g loss={} g loss={} d loss={} acc={} simi loss={} db loss={}".format(epoch, test_total_loss, test_g_loss, test_d_loss, test_acc, simi_loss, db_loss))
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
        # self.model_g(self.graph[0], self.graph[1:])
        self.model_g.load_state_dict(torch.load(self.best_path))
        self.model_g.eval()
        encoded, decoded = self.model_g(self.graph[0], self.graph[1:])
        clusters = self.Cluster.fit_predict(encoded.cpu().detach().numpy())
        loc_index = 0
        for i in range(len(self.expr_ad_list)):
            if key in self.expr_ad_list[i].obs.columns:
                self.expr_ad_list[i].obs = self.expr_ad_list[i].obs.drop(columns=key)
            self.expr_ad_list[i].obs[key] = clusters[loc_index:loc_index+self.expr_ad_list[i].shape[0]]
            self.expr_ad_list[i].obs[key] = pd.Categorical(self.expr_ad_list[i].obs[key])
            self.expr_ad_list[i].uns[f'{key}_colors'] = [color_map.loc[c,'color'] for c in self.expr_ad_list[i].obs['spatial_domain'].cat.categories]
            loc_index += self.expr_ad_list[i].shape[0]
