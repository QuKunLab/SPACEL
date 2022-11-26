import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2, l2
from .kegra.gnn import GraphConvolution
from .kegra.gnn_utils import *
from scipy.sparse import coo_matrix

def scale(z, epsilon=1e-7):
    zmax = tf.reduce_max(z,axis=1, keepdims=True)
    zmin = tf.reduce_min(z,axis=1, keepdims=True)
    z_std = tf.math.divide_no_nan(z - zmin,(zmax - zmin))
    return z_std

def l2_activation(x):
    x = scale(x)
    x = tf.nn.l2_normalize(x,axis=1)
    return x

def kl_divergence(y_true, y_pred, axis=0):
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), np.inf)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.math.divide_no_nan(y_true, tf.reduce_sum(y_true, axis=axis, keepdims=True))
    y_pred = tf.math.divide_no_nan(y_pred, tf.reduce_sum(y_pred, axis=axis, keepdims=True))
    y_true = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1)
    return tf.reduce_mean(tf.multiply(y_true, tf.math.log(tf.math.divide_no_nan(y_true,y_pred))), axis=axis)

def get_GNN_inputs(celltype_ad_list):
    print('Generating GNN inputs...')
    A_list = []
    X_list = []
    for celltype_ad in celltype_ad_list:
        X_tmp = np.matrix(celltype_ad.X,dtype='float32')
        X_list.append(X_tmp)
        A_list.append(coo_matrix(celltype_ad.obsp['spatial_distances'],dtype='float32'))

    X_raw = np.concatenate(X_list)
    element_num = 0
    for A_tmp in A_list:
        element_num += A_tmp.shape[0]
    A = np.zeros((element_num,element_num),dtype='float32')
    loc_index = 0
    class_index = 0
    slice_class = []
    for A_tmp in A_list:
        A[loc_index:loc_index+A_tmp.shape[0],loc_index:loc_index+A_tmp.shape[0]] = A_tmp.toarray()
        slice_class = slice_class + [class_index]*A_tmp.shape[0]
        loc_index += A_tmp.shape[0]
        class_index += 1
    A = coo_matrix(A,dtype='float32')
    nb_mask = np.array(np.where((A>0).todense()))
    slice_class_onehot = tf.one_hot(slice_class,depth=max(slice_class)+1)
    return X_raw,A,nb_mask,slice_class_onehot

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = coo_matrix(X)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def get_GNN_kernel(X,A,k=2):
    # Normalize X

    SYM_NORM=False
    X_filtered = (X-X.mean(0))/X.std(0)
    X_filtered = tf.convert_to_tensor(X_filtered,dtype='float32')
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    L = normalized_laplacian(A, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, k)
    T_k_coo = []
    for g in T_k:
        T_k_coo.append(tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(g)))
    support = k + 1
    graph = [X_filtered]+T_k_coo
    G = [Input(shape=(None,), sparse=True) for _ in range(support)]
    return X_filtered, graph, G, support

def split_train_test_data(X,train_prop):
    rand_idx = np.random.permutation(X.shape[0])
    train_idx = rand_idx[:int(len(rand_idx)*train_prop)]
    test_idx = rand_idx[int(len(rand_idx)*train_prop):]
    train_mask = sample_mask(train_idx,X.shape[0])
    test_mask = sample_mask(test_idx,X.shape[0])
    return train_mask, test_mask

class GraphConvolutionModel(tf.keras.models.Model):
    def __init__(self,feature_dims,G,support,latent_dims=8,hidden_dims=64,dropout=0.8):
        super(GraphConvolutionModel, self).__init__()
        self.G = G
        self.support = support
        self.latent_dims=latent_dims
        self.hidden_dims=hidden_dims
        self.feature_dims = feature_dims
        self.dropout = dropout
        self.encoder=self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        X_in = Input(shape=(self.feature_dims,))
        H = Dropout(self.dropout)(X_in)
        H = GraphConvolution(self.hidden_dims, self.support, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=l1_l2(), use_bias=False)([H]+self.G)
        H = Dropout(self.dropout)(H)
        Y = GraphConvolution(self.latent_dims, self.support, activation=l2_activation, kernel_initializer=tf.keras.initializers.GlorotUniform(), use_bias=False)([H]+self.G)
        return Model(inputs=[X_in]+self.G, outputs=Y)

    def build_decoder(self):
        X_in = Input(shape=(self.latent_dims,))
        H = Dropout(self.dropout)(X_in)
        H = GraphConvolution(self.hidden_dims, self.support, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=l1_l2(), use_bias=False)([H]+self.G)
        H = Dropout(self.dropout)(H)
        Y = GraphConvolution(self.feature_dims, self.support, activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform(), use_bias=False)([H]+self.G)
        return Model(inputs=[X_in]+self.G, outputs=Y)

    def call(self,inputs,encode_only=False,decode_only=True):
        encoded = self.encoder(inputs)
        if encode_only:
            return encoded
        else:
            G_encoded = [encoded]+inputs[1:]
            decoded = self.decoder(G_encoded)
            if decode_only:
                return decoded
            else:
                return encoded, decoded

class DiscriminationModel(tf.keras.models.Model):
    def __init__(self,Y,latent_dims=8,hidden_dims=64):
        super(DiscriminationModel, self).__init__()
        self.hidden_dims=hidden_dims
        self.latent_dims = latent_dims
        self.class_num = Y.shape[1]
        self.model=self.build_model()

    def build_model(self):
        X_in = Input(shape=(self.latent_dims,))
        H = Dense(self.hidden_dims, activation=None)(X_in)
        H = BatchNormalization()(H)
        H = tf.keras.layers.LeakyReLU()(H)
        H = Dense(self.hidden_dims, activation=None)(H)
        H = BatchNormalization()(H)
        H = tf.keras.layers.LeakyReLU()(H)
        H = Dropout(0.5)(H)
        Y = Dense(self.class_num, activation='softmax')(H)
        return Model(inputs=X_in, outputs=Y)

    def call(self,x):
        return self.model(x)