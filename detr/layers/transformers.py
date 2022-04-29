from detr_tf.networks.transformer import MultiHeadAttention
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dropout, prefix):
        self.d_model = d_model
        self.nhead = nhead
        
        assert d_model % nhead == 0
        self.d_Head = d_model // nhead
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.prefix = prefix

    # TODO: check whetehr code is valid or not    
    def build(self, input_shapes):
        in_dim = sum([shape[-1] for shape in input_shapes[:3]])

        self.in_proj_weight = self.add_weight(
            name=f'{self.prefix}/in_proj_kernel', shape=(in_dim, self.model_dim),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )
        self.in_proj_bias = self.add_weight(
            name=f'{self.prefix}/in_proj_bias', shape=(in_dim,),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )
        self.out_proj_weight = self.add_weight(
            name=f'{self.prefix}/out_proj_kernel', shape=(self.model_dim, self.model_dim),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )
        self.out_proj_bias = self.add_weight(
            name=f'{self.prefix}/out_proj_bias', shape=(self.model_dim,),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )
    
    # TODO: convert commented pseudo code into proper code
    def call(self, query, key, value, attn_mask, key_padding_mask, need_weights=True, **kwargs):
        qkv = tf.concat([query, key, value], axis=-1)
        qkv_w = qkv @ tf.transpose(self.in_proj_weight, perm=(1, 0)) + self.in_proj_bias
      
  
        # WQ: normalize by sqrt(self.d_head) & reshape (target_len:=tf.shape(query)[0], batch_size:=tf.shape(query)[1] * self.nheads, self.d_head) & transpose 1 0 2
        # WK: reshape (source_len:=tf.shape(key)[0], batch_Size * self.nheads, self.d_head) then transpose 1 0 2
        # WV: reshape (source_len:=tf.shape(key)[0], batch_Size * self.nheads, self.d_head) then transpose 1 0 2
        
        # attn_output_weights = WQ @ tf.transpose(WK, perm=(0, 2, 1))
        if attn_mask is not None:
            attn_output_weights += attn_mask
        # attn_output_weights = self.dropout(softmax(attn_output_weights, axis=-1), **kwargs)
        
        # attn_output = tf.transpose(attn_output_weights @ WV, perm=(1, 0, 2))
        # TODO: understand following code        
        # attn_output = tf.reshape(attn_output, (target_len, batch_size, self.model_dim))
        # attn_output = attn_output @ tf.transpose(self.out_proj_weight, perm=(0, 2, 1)) + self.out_proj_bias
        
        if need_weights:
            attn_output_weights = tf.reshape(attn_output_Weights, (batch_Size, self.num_heads, target_len, source_len))
            avg_weights = tf.reduce_mean(attn_output_weights, axis=1)
            return attn_output, avg_weights
        return attn_oiutput
        
        
        
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward, 
                 dropout, activation, normalize_before, prefix):
        super().__init__(name=prefix)
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, prefix=f'{prefix}/self_attn')
        self.dense1 = tf.keras.layers.Dense(dim_feedforward, activation=activation, name=f'{prefix}/dense1')
        self.dropout = tf.keras.layers.Dropout(dropout, name=f'{prefix}/dropout')
        self.dense2 = tf.keras.layers.Dense(d_model, name=f'{prefix}/dense2')
        
        self.norm1 = tf.keras.layers.LayerNormalization(name=f'{prefix}/norm1')
        self.norm2 = tf.keras.layers.LayerNormalization(name=f'{prefix}/norm2')
        self.dropout1 = tf.keras.layers.Dropout(dropout, name=f'{prefix}/dropout1')
        self.dropout2 = tf.keras.layers.Dropout(dropout, name=f'{prefix}/dropout2')
        self.call = self.call_pre if normalize_before else self.call_post
        
    def with_pos_embeding(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def call_post(self, src, src_mask, src_key_padding_mask, pos):
        q = k = self.with_pos_embeding(src, pos)
        # TODO: check output
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.dense2(self.dropout(self.dense1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
    def call_pre(self, src, src_mask, src_key_padding_mask, pos):
        src = self.norm1(src)
        q = k = self.with_pos_embeding(src, pos)
        # TODO: check output
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src2 = self.dense2(self.dropout(self.dense1(src)))
        src = src + self.dropout2(src2)
        return src
        
class TransformerEncoder(tf.keras.layers.Layers):
    def __init__(self, d_model, nhead, num_layers,
                 dim_feedforward, dropout, activation,
                 normalize_before, prefix=''):
        super().__init__(name=prefix)
        self.layers = [TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                               dropout, activation, normalize_before,
                                               prefix=f'{prefix}/enc_layers{i_layer}')
                       for i_layer in range(num_layers)]
        self.num_layers = num_layers
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f'{prefix}/norm') if normalize_before else None

    def __call__(self, src, mask, src_key_padding_mask, pos):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        
        if self.norm is not None:
            output = self.norm(output)
        return output




class Transformer(tf.keras.layers.Layer):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, prefix='', **kwargs):
        super().__init__(name=prefix)
        self.encoder = TransformerEncoder(d_model, nhead, num_encoder_layers,
                                          dim_feedforward, dropout,
                                          activation, normalize_before, prefix=f'{prefix}/encoder')
        self.decoder = TransformerDecoder(d_model, nhead, num_decoder_layers,
                                          dim_feedforward, dropout,
                                          activation, normalize_before,
                                          return_intermediate=return_intermediate_dec,
                                          prefix='')




