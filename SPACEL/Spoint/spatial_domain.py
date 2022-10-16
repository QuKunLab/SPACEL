import tensorflow as tf
import tensorflow_addons as tfa

def add_conv_block(model,filters,kernel_size,padding='same',transpose=False,*args, **kwargs):
    if not transpose:
        model.add(tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,padding=padding,*args, **kwargs))
    else:
        model.add(tf.keras.layers.Conv2DTranspose(filters=filters,kernel_size=kernel_size,padding=padding,*args, **kwargs))
    model.add(tf.keras.layers.LayerNormalization(axis=[1,2]))
#     model.add(tf.keras.layers.AveragePooling2D(pool_size=2,padding=padding))
    model.add(tf.keras.layers.LeakyReLU())
    
def kl_divergence(y_true, y_pred):
    y_true = tf.reshape(y_true,-1)
    y_pred = tf.reshape(y_pred,-1)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), np.inf)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.math.divide_no_nan(y_true, tf.reduce_sum(y_true, axis=-1, keepdims=True))
    y_pred = tf.math.divide_no_nan(y_pred, tf.reduce_sum(y_pred, axis=-1, keepdims=True))
    y_true = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1)
    return tf.reduce_sum(tf.multiply(y_true, tf.math.log(tf.math.divide_no_nan(y_true,y_pred))), axis=-1)

class ConvModel(tf.keras.models.Model):
    def __init__(
        self,
        exp_length,
        density_length,
        latent_dims,
        kernel_size,
    ):
        super(ConvModel, self).__init__()
#         self.exp_encoder = self.build_exp_encoder(kernel_size)
#         self.density_encoder = self.build_density_encoder(kernel_size)
#         self.concat_encoder = self.build_concat_encoder(latent_dims,kernel_size)
        self.encoder = self.build_encoder(latent_dims,kernel_size)
        self.decoder = self.build_decoder(exp_length+density_length,kernel_size)
        
        
    def build_exp_encoder(self,kernel_size):
        model = tf.keras.Sequential()
        add_conv_block(model,filters=1,kernel_size=kernel_size)
        return model
    
    def build_density_encoder(self,kernel_size):
        model = tf.keras.Sequential()
        add_conv_block(model,filters=1,kernel_size=kernel_size)
        return model
    
    def build_concat_encoder(self,latent_dims,kernel_size):
        model = tf.keras.Sequential()
        add_conv_block(model,filters=128,kernel_size=kernel_size)
        add_conv_block(model,filters=64,kernel_size=kernel_size)
        add_conv_block(model,filters=32,kernel_size=kernel_size)
        model.add(tf.keras.layers.Conv2D(filters=latent_dims,kernel_size=kernel_size,padding='same'))
        return model
    
    def build_encoder(self,latent_dims,kernel_size):
        model = tf.keras.Sequential()
        add_conv_block(model,filters=256,kernel_size=kernel_size,kernel_regularizer='l1_l2')
        add_conv_block(model,filters=256,kernel_size=kernel_size,kernel_regularizer='l1_l2')
        add_conv_block(model,filters=256,kernel_size=kernel_size,kernel_regularizer='l1_l2')
        model.add(tf.keras.layers.Dropout(0.8))
        model.add(tf.keras.layers.Conv2D(filters=latent_dims,kernel_size=kernel_size,padding='same',activation='sigmoid'))
        return model
    
    def build_decoder(self,rec_dims,kernel_size):
        model = tf.keras.Sequential()
        add_conv_block(model,filters=256,kernel_size=kernel_size,transpose=True,kernel_regularizer='l1_l2')
        add_conv_block(model,filters=256,kernel_size=kernel_size,transpose=True,kernel_regularizer='l1_l2')
        add_conv_block(model,filters=256,kernel_size=kernel_size,transpose=True,kernel_regularizer='l1_l2')
        model.add(tf.keras.layers.Dropout(0.8))
        model.add(tf.keras.layers.Conv2DTranspose(filters=rec_dims,kernel_size=kernel_size,padding='same'))
        return model
    
#     def call(self,exp,celltype_density):
#         exp_encoded = self.exp_encoder(exp)
#         density_encoded = self.density_encoder(celltype_density)
#         concated = tf.concat([exp_encoded,density_encoded],axis=-1)
#         concat_encoded = self.concat_encoder(concated)
#         decoded = self.decoder(concat_encoded)
#         return decoded

    def call(self,exp,celltype_density,encode_only=False):
        concated = tf.concat([exp,celltype_density],axis=-1)
        encoded = self.encoder(concated)
        if encode_only:
            return encoded
        decoded = self.decoder(encoded)
        exp_rec = decoded[:,:,:,:exp.shape[-1]]
        density_rec = decoded[:,:,:,exp.shape[-1]:]
        return exp_rec, density_rec
    
#     def encode(self,exp,celltype_density):
#         exp_encoded = self.exp_encoder(exp)
#         density_encoded = self.density_encoder(celltype_density)
#         concated = tf.concat([exp_encoded,density_encoded],axis=-1)
#         concat_encoded = self.concat_encoder(concated)
#         return concat_encoded

#     def encode(self,exp,celltype_density):
#         exp_encoded = self.exp_encoder(exp)
#         density_encoded = self.density_encoder(celltype_density)
#         concated = tf.concat([exp_encoded,density_encoded],axis=-1)
#         concat_encoded = self.concat_encoder(concated)
#         return concat_encoded