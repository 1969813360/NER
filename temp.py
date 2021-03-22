from Classes.NERController import NERController

if __name__ == '__main__':
    MyController=NERController(
        charWindowSize=4,#共现窗口大小（同时观测该字符前后多少个字符）
        maxSentenceLenth=50,#文章的最大长度  cluener:50,ccf:100 ,weibo:50
        lr=0.001,#学习率
        epoch=30,#迭代多少轮
        batchSize=16,#batchsize
        emb_dim=64,#embedding维度
        hidden_dim=128,#隐藏层维度
        dropout=0.5,#dropout
        datatype='weibo',#0:对CCF数据集训练，1：对cluener数据集训练，2：对weibo数据集训练
        embedding_model_type=0,#使用什么预训练模型，0：word2vec；1：Bert
        old_or_new='old',
        is_noised='noised',
        which_epoch_to_test=5
    )
    MyController.train()