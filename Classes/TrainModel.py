import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from .Layers import Crf_layer,Bi_LSTM
import tensorflow_addons as tfa
#训练模型类,继承Model类
class TrainModel(Model):
    def __init__(self,hidden_dim,dropout,category_num,istraining):
        super(TrainModel, self).__init__()
        self.bi_Lstm=Bi_LSTM(hidden_dim,dropout)
        self.reshape=Dense(category_num,trainable=istraining)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.crf=Crf_layer(category_num)

    def call(self, datas,labels,lens,istraining):
        x=self.bi_Lstm(datas)
        x=self.dropout(x,istraining)
        x=self.reshape(x)
        if istraining:
            predict,loss=self.crf(x,labels,lens)
            return predict,loss
        else:
            predict=self.crf(x,labels,lens)
            return predict