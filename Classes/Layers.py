# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

w_path = "./model/data/w.pkl"#存储CRF层的概率转移参数矩阵

class Bi_LSTM(layers.Layer):

    def __init__(self, hidden_dim,dropout):
        super().__init__(self)
        #第一层biLSTM
        self.fw_lstm1 = layers.LSTM(hidden_dim, return_sequences=True, go_backwards=False,
                                    dropout=dropout, name="fwd_lstm1")
        self.bw_lstm1 = layers.LSTM(hidden_dim, return_sequences=True, go_backwards=True,
                                    dropout=dropout, name="bwd_lstm1")

        self.bilstm1 = layers.Bidirectional(merge_mode="ave", layer=self.fw_lstm1, backward_layer=self.bw_lstm1,
                                            name="bilstm1")

        #第二层biLSTM
        self.fw_lstm2 = layers.LSTM(hidden_dim, return_sequences=True, go_backwards=False,
                                    dropout=dropout, name="fwd_lstm2")
        self.bw_lstm2 = layers.LSTM(hidden_dim, return_sequences=True, go_backwards=True,
                                    dropout=dropout, name="bwd_lstm2")
        self.bilstm2 = layers.Bidirectional(merge_mode="ave", layer=self.fw_lstm2, backward_layer=self.bw_lstm2,
                                            name="bilstm2")

    def call(self, inputs,):
        x = self.bilstm1(inputs)
        y = self.bilstm2(x)
        return y

# 自注意力机制层
class Self_Attention_Layer(layers.Layer):
    def __init__(self, para):
        super().__init__(self)
        self.para = para
        self.dense_Q = layers.Dense(self.para.attention_size, use_bias=False, trainable=True,
                                    kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.dense_K = layers.Dense(self.para.attention_size, use_bias=False, trainable=True,
                                    kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.dense_V = layers.Dense(self.para.attention_size, use_bias=False, trainable=True,
                                    kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.dropout = layers.Dropout(self.para.attenion_drop_rate, name='attenion_drop')
        self.softmax = layers.Softmax()

    def call(self, inputs, sen_len):
        # 就算QKV
        Q = self.dense_Q(inputs)
        K = self.dense_K(inputs)
        V = self.dense_V(inputs)

        # 下面开始做注意力机制,如果使用mask操作，还要用到句子的长度,不使用mask操作会简单很多
        QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # 现在QK的大小是[batch_size,max_sen_len,max_sen_len]
        if self.para.is_mask:
            # 接下来实现带有mask操作的自注意力机制,-，现在再次尝试
            mask = tf.sequence_mask(sen_len, maxlen=self.para.max_sen_len)
            mask = tf.expand_dims(mask, 1)  # mask主要是将填充的地方的权值设置的非常小，这样在加权的时候就会是填充的单词起到作用了
            mask = tf.tile(mask, [1, tf.shape(QK)[1], 1])  # 现在有了mask矩阵，下面开始将pading的单词的权重是设置的很小
            padding_val = -2 ** 32
            QK = tf.where(mask, QK, tf.ones_like(QK) * padding_val) / tf.sqrt(
                tf.cast(self.para.hidden_dim, dtype=tf.float32))  # 采用的是缩放的点积
            QK = self.softmax(QK)
            Z = tf.matmul(QK, V)
        else:
            # 不使用mask操作，还要有个缩放因子
            QK = self.softmax(QK / tf.sqrt(self.para.hidden_dim))
            # softmax之后就是加权求和输出z，很简单的矩阵乘法
            Z = tf.matmul(QK, V)  # 使用这个矩阵乘法之后，默认在最后两个维度进行做乘法，也就是加权求和了
        return Z

# 加性注意力机制,该注意力机制好像是首先应用在seq2seq模型里面的，需要使用到编码器和解码器这两个部分的向量，
# 但是对于这个LSTM+attention+crf实现命名实体识别模型的，由于没有解码器，因为隐向量只有一部分，如何做到
class additive_attention_layer(layers.Layer):
    '''
    至于为什么要这么去实现加性attention机制，我也不清楚。因为深度学习本来就是解释性特别的低
    我只是在想如何计算一组权重，这种权重可以根据不同的计算方式得到，但是到底是好是坏谁也不知道，
    因而我就根据网上的一些文章，自己去尝试实现一下。对不对估计还需要大佬的指教,模型的好坏只能靠实验结果确定
    '''

    def __init__(self, para):
        super().__init__(self)
        self.para = para
        self.dense = layers.Dense(self.para.attention_size, trainable=True, activation='tanh')  # 这个是需要一个激活函数的
        self.dropout = layers.Dropout(rate=self.para.attenion_drop_rate)
        self.softmax = layers.Softmax()

    def build(self, input_shape):
        # 我想使用这个权值矩阵将经过全连接层作用之后的输出的大小[batch_size,max_len,attention_size]
        # 调正为[batch_size,maxlen,maxlen],就和自注意力机制层是一样的
        self.attention_u = self.add_weight(name='atten_u', shape=(self.para.attention_size, self.para.max_sen_len),
                                           initializer=tf.random_uniform_initializer(), dtype="float32", trainable=True)
        super.build(input_shape)

    def call(self, inputs, sen_len):
        '''
        Parameters
        ----------
        inputs : tensor
            循环神经网络的输出.
        sen_len : tensor
            每个batch里面句子的长度，用来实现mask操作.

        Returns
        -------
        返回加权之后的隐向量.

        '''
        alpha = self.dense(inputs)
        if self.para.is_mask:
            alpha = tf.matmul(alpha, self.attention_u)
            mask = tf.sequence_mask(sen_len, maxlen=self.para.max_sen_len)
            mask = tf.expand_dims(mask, 1)
            mask = tf.tile(mask, [1, tf.shape(alpha)[1]], 1)  # [batch_size,maxlen,maxlen]
            padding_val = -2 ** 22
            alpha = tf.where(mask, alpha, tf.ones_like(alpha) * padding_val)
            alpha = self.softmax(alpha)
            Z = tf.matmul(alpha, inputs)
        else:
            alpha = tf.matmul(alpha,
                              self.attention_u)  # 将alpha的大小由[batch_size,max_len,attention_size]调整为大小为[batch_size,maxlen,maxlen]
            alpha = self.softmax(alpha)
            Z = tf.matmul(alpha, inputs)  # 将权值与隐向量相乘做为新的隐向量
        return self.dropout(Z, training=self.para.is_training)

class Crf_layer(layers.Layer):
    def __init__(self,category_num):
        super().__init__(self)
        self.paras=tf.Variable(tf.random.uniform(shape=(category_num, category_num)))


    def call(self, inputs, targets, lens):
        '''
        inputs需调整为[batch_size,maxlen,label_nums]
        '''
        if targets is not None:
            loss,self.paras=tfa.text.crf_log_likelihood(inputs,targets,lens,transition_params=self.paras)
            predict, score = tfa.text.crf_decode(inputs, self.paras, lens)
            loss=tf.reduce_sum(-loss)
            return predict,loss
        else:
            predict, score = tfa.text.crf_decode(inputs, self.paras, lens)
            return predict