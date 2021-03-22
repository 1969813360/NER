import jieba
import tensorflow as tf
import pickle
import os
import torch
import json
import numpy as np
from gensim.models import word2vec,KeyedVectors
from transformers import BertTokenizer,BertModel,BertConfig
from .TrainModel import TrainModel
from sklearn import model_selection,metrics

bertpath=r'./cache/bert/bert-base-chinese-pytorch_model.bin'#Bert预训练模型的存放路径
word2vec_dicpath=r'./cache/word2vec/dic_64.txt'#word2vec训练的词的embedding保存路径
jsonpath=[r'./resource/ccf_14.json',r'./resource/cluener_10.json',r'./resource/cluener_10_test.json']#三个数据集路径
weibopath=r'./resource/new/weibo_4_noised_train.txt'
resultpath=r'./result/'

class NERController:
    def __init__(self,charWindowSize,maxSentenceLenth,lr,epoch,batchSize,
                 emb_dim,hidden_dim,dropout,datatype,embedding_model_type,old_or_new,is_noised,which_epoch_to_test):
        self.charWindowSize=charWindowSize#字符的共现窗口大小，即同时观察当前字符前后多少个字符
        self.maxSentenceLenth=maxSentenceLenth#文章的最大长度
        self.lr=lr#学习率
        self.epoch=epoch#迭代轮数
        self.batchSize=batchSize#一次喂入多少条数据
        self.emb_dir=emb_dim#每个词的embedding维数
        self.hidden_dim=hidden_dim#隐藏层的维度，也即第一层LSTM层的记忆体个数
        self.dropout=dropout
        self.bert_token = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model_config = BertConfig.from_pretrained('bert-base-chinese')
        self.datatype=datatype#数据集类型
        self.embedding_model_type=embedding_model_type#预训练模型是Bert还是word2vec
        self.old_or_new=old_or_new
        self.is_noised=is_noised
        self.which_epoch_to_test=which_epoch_to_test

        # GPU information
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # use GPU with ID=0

        if self.datatype=='ccf':#CCF数据集
            self.ctg_dic=['O', 'B-position', 'M-position', 'E-position', 'B-name', 'M-name', 'E-name',
                     'B-organization', 'M-organization', 'E-organization', 'B-movie', 'M-movie',
                     'E-movie', 'B-email', 'M-email', 'E-email', 'B-mobile', 'M-mobile', 'E-mobile', 'B-company',
                     'M-company', 'E-company', 'B-book', 'M-book', 'E-book', 'B-QQ', 'M-QQ', 'E-QQ',
                     'B-scene', 'M-scene', 'E-scene', 'B-address', 'M-address', 'E-address',
                      'B-game', 'M-game', 'E-game', 'B-government', 'M-government',
                     'E-government', 'B-vx', 'M-vx', 'E-vx', 'H']
            self.category_num=44
            if self.is_noised=='noised':
                self.trainfilepath=r'./resource/'+self.old_or_new+'/ccf_14_noised_train.txt'#原文本路径
                self.testfilepath=r'./resource/'+self.old_or_new+'/ccf_14_noised_test.txt'#原文本路径
            else:
                self.trainfilepath = r'./resource/' + self.old_or_new + '/ccf_14_train.txt'  # 原文本路径
                self.testfilepath = r'./resource/' + self.old_or_new + '/ccf_14_test.txt'  # 原文本路径
            self.datapath=r'./cache/data/ccf/'#喂进模型的数据路径
            self.model_variable = r'../NER/cache/variable/ccf/'#模型的可训练参数保存路径
        elif self.datatype=='cluener':#10分类数据集
            self.ctg_dic=['O','B-company','M-company','E-company',  'B-name', 'M-name', 'E-name',
                     'B-email', 'M-email', 'E-email', 'B-mobile', 'M-mobile', 'E-mobile', 'B-game', 'M-game',
                     'E-game', 'B-QQ', 'M-QQ', 'E-QQ', 'B-organization', 'M-organization', 'E-organization',
                     'B-movie', 'M-movie', 'E-movie', 'B-position', 'M-position', 'E-position',
                     'B-address', 'M-address', 'E-address', 'B-government', 'M-government', 'E-government',
                     'B-scene', 'M-scene', 'E-scene',  'B-book', 'M-book', 'E-book', 'H']
            self.category_num = 41
            if self.is_noised == 'noised':
                self.trainfilepath = r'./resource/' + self.old_or_new + '/cluener_10_noised_train.txt' # 原文本路径
                self.testfilepath = r'./resource/' + self.old_or_new + '/cluener_10_noised_test.txt'  # 原文本路径
            else:
                self.trainfilepath = r'./resource/' + self.old_or_new + '/cluener_10_train.txt' # 原文本路径
                self.testfilepath = r'./resource/' + self.old_or_new + '/cluener_10_test.txt'  # 原文本路径
            self.datapath = r'./cache/data/cluener/'#喂进模型的数据路径
            self.model_variable = r'../NER/cache/variable/cluener/'#模型的可训练参数保存路径
        elif self.datatype=='weibo':#微博数据集
            self.ctg_dic=['O', 'B-email', 'M-email', 'E-email', 'B-mobile', 'M-mobile', 'E-mobile', 'B-QQ', 'M-QQ', 'E-QQ',
                     'B-GPE', 'M-GPE', 'E-GPE', 'B-PER', 'M-PER', 'E-PER', 'B-ORG', 'M-ORG', 'E-ORG', 'B-LOC',
                     'M-LOC', 'E-LOC',  'S-PER', 'S-GPE', 'S-LOC', 'H']
            self.category_num = 26
            if self.is_noised == 'noised':
                self.trainfilepath = r'./resource/'+self.old_or_new+'/weibo_4_noised_train.txt'#原文本路径
                self.testfilepath = r'./resource/' + self.old_or_new + '/weibo_4_noised_test.txt'  # 原文本路径
            else:
                self.trainfilepath = r'./resource/'+self.old_or_new+'/weibo_4_train.txt'#原文本路径
                self.testfilepath = r'./resource/' + self.old_or_new + '/weibo_4_test.txt'  # 原文本路径
            self.datapath = r'./cache/data/weibo/'#喂进模型的数据路径
            self.model_variable = r'./cache/variable/weibo/'#模型的可训练参数保存路径

    #开始训练模型
    def train(self):
        x,y,lens=self.get_splited_data('train')
        print("已成功获取训练集！")


        x_train=tf.cast(x,dtype=tf.float32)
        y_train=tf.cast(y,dtype=tf.int32)
        lens_train = tf.cast(lens, dtype=tf.int32)

        train_data_bitchs = tf.data.Dataset.from_tensor_slices((x_train, y_train, lens_train)).batch(self.batchSize,
                                                                                                     drop_remainder=True)

        print("开始训练模型！")
        # 指定训练模型
        model=TrainModel(self.hidden_dim,self.dropout,self.category_num,True)
        if self.is_noised=='noised':
            modelFile = self.model_variable +'noised/'+self.old_or_new+"/"+self.old_or_new+ "_model.ckpt"
        else:
            modelFile = self.model_variable + self.old_or_new + "/" + self.old_or_new + "_model.ckpt"
        if os.path.exists(modelFile + '.index'):
            print('---------------正在加载训练好的模型参数-----------------')
            model.load_weights(modelFile)

        #优化器
        optimizer = tf.keras.optimizers.Adam(self.lr)


        def get_predict_loss(x, y, lens):
            with tf.GradientTape() as tape:
                predict,loss= model(x, y, lens,True)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return predict,loss


        def getAcc(predict,lables,lens):
            total=tf.reduce_sum(lens)
            predict = predict.__array__()
            lables=lables.__array__()
            lens = lens.__array__()
            right_num=0
            for i in range(len(lens)):  # 文本下标
                for j in range(lens[i]):
                    if lables[i][j]==predict[i][j]:
                        right_num+=1
            acc=right_num/total
            return acc

        maxAcc=0.0
        minLoss=0.0
        for epoch in range(self.epoch):
            Acc=[]
            Loss=[]
            for step, (x_train, y_train,lens) in enumerate(train_data_bitchs):  # 遍历批次
                predict,loss=get_predict_loss(x_train,y_train,lens)
                acc=getAcc(predict,y_train,lens)
                Acc.append(acc)
                Loss.append(abs(loss))
                if step%50==0:
                    print("    第%d epoch->第%d step的loss：%.6f    acc:%.6f" % (epoch,step,float(loss),float(acc)))

            # 保存模型
            ave_acc=np.mean(Acc)
            ave_loss=np.mean(Loss)
            if ave_acc>maxAcc:#当前轮的平均ACC大于maxAcc,则将参数保存
                maxAcc=ave_acc
                model.save_weights(modelFile,True)
            elif ave_acc==maxAcc and ave_loss<minLoss:
                minLoss=ave_loss
                model.save_weights(modelFile, True)

            if (epoch+1) % self.which_epoch_to_test == 0:
                self.predict(epoch+1)

        print("模型训练完毕！参数已保存至:  %s!" % modelFile)

    def getDic(self):
        print('dic_path is : ', word2vec_dicpath)
        if os.path.exists(word2vec_dicpath) ==False:#没有发现词典
            print("未在本地发现词典，即将开始自训练，请稍等...")
            self.dic=self.generateDic()
        else:
            self.dic=KeyedVectors.load_word2vec_format(word2vec_dicpath, binary=False)
            print("已从本地文件中加载词典！")

    #根据训练集和测试题自己训练，使用word2vec训练词的embedding
    def generateDic(self):
        print("开始使用word2vec训练词典，请稍等...")
        dic=[]#存储所有词、字

        for path in jsonpath:
            with open(path,'r',encoding='utf-8') as fp:
                lines=fp.readlines()
            for line in lines:
                json_data = json.loads(line)['text']
                words = jieba.cut(json_data, cut_all=True)
                words = list(words)
                words.extend(list(json_data))
                dic.append(words)

        #weibo
        with open(weibopath,'r',encoding='utf-8') as fp:
            lines=fp.readlines()
        text=''
        for line in lines:
            if line == '\n':
                words = jieba.cut(text, cut_all=True)
                words = list(words)
                words.extend(list(text))
                dic.append(words)
                text=''
            else:
                text+=line[0]
        #对字典训练，得到embedding
        emb=word2vec.Word2Vec(dic, sg=1, iter=2000, size=self.emb_dir, window=5, min_count=1, negative=3, sample=0.001, hs=1, workers=16)
        emb.wv.save_word2vec_format(word2vec_dicpath,binary=False)
        print("词典训练完毕，已保存在本地%s" % word2vec_dicpath)
        return emb

    #根据集合B,M,E里的单词，得到该集合的embedding表示
    def generateEmb(self,S):
        #每个词的embedding对位相加取平均
        result=tf.zeros(self.emb_dir)
        if self.embedding_model_type==0:#使用word2vec预训练模型
            for item in S:
                if item in self.dic.wv.index2word:  # 该词存在字典中
                    temp = tf.constant(self.dic[item], dtype=tf.float32)
                else:  # 不在字典中则随机初始化一个
                    print("警告：【%s】不在词典中，已随机赋值。稍后请更新字典" % item)
                    temp = tf.random.uniform([self.emb_dir], minval=-1, maxval=1)
                    self.dic.wv[item]=temp.numpy()
                result = tf.add(result, temp)
            if len(S) != 0:  # 判断集合内元素的个数是否为0，否则0作为除数报错
                result = result / len(S)
        else:#使用Bert预训练模型
            for item in S:
                try:
                    index = self.bert_token.convert_tokens_to_ids(item)
                    bert_model = BertModel.from_pretrained(bertpath, config=self.model_config)
                    batch_data = torch.Tensor(index).long().view((-1, 1))
                    temp=tf.constant(bert_model(batch_data)[-1][0].detach().numpy(),dtype=tf.float32)
                except IndexError:
                    print("警告：【%s】不在Bert预训练模型的词典中，已随机赋值。" % item)
                    temp=tf.random.uniform([self.emb_dir],minval=-1,maxval=1)
                result = tf.add(result, temp)
            if len(S) !=0:#判断集合内元素的个数是否为0，否则0作为除数报错
                result = result / len(S)
        return result.__array__()

    #根据字符的{B,M,E}向量,从而得到每个字符的embedding
    def getEmb(self,B,M,E,char):
        result=[]
        if self.embedding_model_type==0:
            if char in self.dic.wv.index2word:  # 字典中有该字符的embedding
                x1 = tf.constant(self.dic[char], dtype=tf.float32)
            else:
                print("警告：【%s】不在词典中，已随机赋值。稍后请更新字典" % char)
                x1 = tf.random.uniform([self.emb_dir], minval=-1, maxval=1)
                self.dic.wv[char] = x1.numpy()
        else:
            try:
                index = self.bert_token.convert_tokens_to_ids(char)
                bert_model = BertModel.from_pretrained(bertpath, config=self.model_config)
                batch_data = torch.Tensor(index).long().view((-1, 1))
                x1 = tf.constant(bert_model(batch_data)[-1][0].detach().numpy(), dtype=tf.float32)
            except IndexError:
                print("警告：【%s】不在Bert预训练模型的词典中，已随机赋值。" % char)
                x1 = tf.random.uniform([self.emb_dir], minval=-1, maxval=1)
        result.extend(x1.__array__())
        result.extend(self.generateEmb(B))
        result.extend(self.generateEmb(M))
        result.extend(self.generateEmb(E))
        return result

    #根据一篇文章，获取文本中每个字符的{B,M,E,S},从而得到每个字符的embedding,将其拼接成该文章的embedding矩阵
    def generateCharEmb(self,text):
        result=[]#存储每个字符的embedding
        size=len(text)
        for i in range(size):#遍历每个字符
            char=text[i]
            B=[]
            M=[]
            E=[]
            beginIndex=i-self.charWindowSize
            endIndex=i+self.charWindowSize
            if beginIndex<0:
                beginIndex=0
            if endIndex>size:
                endIndex=size
            #每次只关注这个字符的前后charWindowSize个位置的字符，也就是说默认最长的词长度为2*charWindowSize+1
            words = jieba.cut(text[beginIndex:endIndex], cut_all=True)
            words=[word for word in words if len(word)>1]#去除分词中的单字
            for word in words:
                if char in word:
                    if char==word[0]:#该字符出现在单词首位
                        B.append(word)
                    elif char==word[-1]:#该字符出现在单词末位
                        E.append(word)
                    else:M.append(word)#该字符出现在单词中间
            #根据字符的{B,M,E,S},从而得到每个字符的embedding
            result.append(self.getEmb(B,M,E,char))
        addTensor=[0 for i in range(4*self.emb_dir)]
        for add in range(self.maxSentenceLenth-size):
            result.append(addTensor)
        return result

    #将类别转为数字
    def ctgtonum(self,category):
        return self.ctg_dic.index(category)

    #将数字还原为类别
    def decodeCtg(self,ctg):
        return self.ctg_dic[ctg]

    #从文件中读取数据，并处理成模型可读取的数据
    def get_Input_Data(self,filepath):
        x_train=[]
        y_train=[]
        lens=[]
        print('data_path is : ',filepath)
        with open(filepath,"r",encoding="utf-8") as fp:
            lines=fp.readlines()
        text=""
        temp=[]
        for line in lines:
            if line =="\n":
                if len(temp)>self.maxSentenceLenth:
                    size=len(temp)//self.maxSentenceLenth
                    remainder=len(temp)%self.maxSentenceLenth
                    for page in range(size):
                        x_train.append(self.generateCharEmb(text[self.maxSentenceLenth*page:self.maxSentenceLenth*(page+1)]))
                        y_train.append(temp[self.maxSentenceLenth*page:self.maxSentenceLenth*(page+1)])
                        lens.append(self.maxSentenceLenth)
                    if remainder!=0:
                        x_train.append(self.generateCharEmb(text[self.maxSentenceLenth*size:]))
                        add_y=[0 for index in range(self.maxSentenceLenth-remainder)]
                        tempy=temp[self.maxSentenceLenth*size:]
                        tempy.extend(add_y)
                        y_train.append(tempy)
                        lens.append(remainder)
                else:
                    x_train.append(self.generateCharEmb(text))
                    add_y = [0 for index in range(self.maxSentenceLenth - len(temp))]
                    temp.extend(add_y)
                    y_train.append(temp)
                    lens.append(len(text))
                temp=[]
                text=""
            else:
                line=line.rstrip()
                line = line.split(" ")
                text=text+line[0]
                temp.append(self.ctgtonum(line[1]))
        return x_train,y_train,lens

    def get_splited_data(self,train_or_test):
        if self.embedding_model_type==0:
            filename='_word2vec.pkl'
        else:
            filename='_bert.pkl'
        tempfilepath=''
        if self.is_noised=='noised':
            tempfilepath='noised_'
        print('x_path is :',self.datapath + 'x_'+train_or_test+'_' + tempfilepath + self.old_or_new + filename)
        if os.path.exists(self.datapath + 'x_'+train_or_test+'_' + tempfilepath + self.old_or_new + filename)==False:#文件不存在
            print("喂入模型的数据还未处理好，请稍等几分钟...")
            self.getDic()
            if train_or_test=='train':
                x,y,lens=self.get_Input_Data(self.trainfilepath)
            else:
                x,y,lens=self.get_Input_Data(self.testfilepath)
            self.dic.wv.save_word2vec_format(word2vec_dicpath,binary=False)#

            x = tf.cast(x, dtype=tf.float32)
            y = tf.cast(y, dtype=tf.int32)
            lens = tf.cast(lens, dtype=tf.int32)

            with open(self.datapath + 'x_'+train_or_test+'_' + tempfilepath + self.old_or_new+filename, "wb") as fp:
                pickle.dump(x, fp)
            with open(self.datapath + 'y_'+train_or_test+'_' + tempfilepath + self.old_or_new+filename, "wb") as fp:
                pickle.dump(y, fp)
            with open(self.datapath + 'lens_'+train_or_test+'_' + tempfilepath + self.old_or_new+filename, "wb") as fp:
                pickle.dump(lens, fp)
            print("喂入模型的数据终于处理好了，已写入指定文件，下次将直接从文件载入数据！")
        else:
            print("本地发现已处理好的x、y、lens。正在载入...")
            with open(self.datapath+'x_'+train_or_test+'_' + tempfilepath + self.old_or_new+filename, "rb") as fp:
                x=pickle.load(fp)
            with open(self.datapath+'y_'+train_or_test+'_' + tempfilepath + self.old_or_new+filename, "rb") as fp:
                y=pickle.load(fp)
            with open(self.datapath+'lens_'+train_or_test+'_' + tempfilepath + self.old_or_new+filename, "rb") as fp:
                lens=pickle.load(fp)
            print("模型数据加载完毕！")
        return x,y,lens

    #处理预测结果
    def get_list(self,data,lens):
        result=[]
        data=data.__array__()
        lens=lens.__array__()
        for i in range(len(lens)):#文本下标
            for j in range(lens[i]):
                result.append(self.decodeCtg(data[i][j]))
        return result

    def predict(self,epoch):
        x, y, lens = self.get_splited_data('test')
        print("已成功获取训练集！")

        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.int32)
        lens = tf.cast(lens, dtype=tf.int32)

        test_data_bitchs = tf.data.Dataset.from_tensor_slices((x, y, lens)).batch(self.batchSize,drop_remainder=True)

        print('-------------------------------现在开始测试...---------------------------')
        # 指定训练模型
        test_model = TrainModel(self.hidden_dim, self.dropout, self.category_num, False)

        if self.is_noised == 'noised':
            modelFile = self.model_variable + 'noised/' + self.old_or_new + "/" + self.old_or_new + "_model.ckpt"
        else:
            modelFile = self.model_variable + self.old_or_new + "/" + self.old_or_new + "_model.ckpt"
        print('modelfile is :',modelFile)
        if os.path.exists(modelFile + '.index'):
            print('---------------正在加载训练好的模型参数-----------------')
            test_model.load_weights(modelFile)

        predict_list=[]
        label_list=[]
        for step, (x_test, y_test, lens) in enumerate(test_data_bitchs):  # 遍历批次
            predict=test_model(x_test,None,lens,False)
            predict_list.extend(self.get_list(predict, lens))
            label_list.extend(self.get_list(y_test,lens))

        report=metrics.classification_report(predict_list,label_list)#打印测试报告

        temppath = ''
        if self.is_noised=='noised':
            temppath='noised_'
        with open(resultpath+self.datatype+'_'+temppath+self.old_or_new+'_report.txt', 'a', encoding='utf-8') as fp:
            fp.write('epoch '+str(epoch)+':\n')
            a,b,c,d=self.get_f1(predict_list,label_list)
            fp.write('全部实体的F1值为：\n'+str(a)+"\n"+str(b)+"\n"+str(c)+"\n"+str(d)+"\n")
            fp.write(report)
            fp.write('\n**********************************************************************************\n')
        if epoch%self.which_epoch_to_test==0:
            self.print_txt_result(epoch,predict_list,label_list)

    def print_txt_result(self,epoch,predict_list,label_list):
        with open(self.testfilepath,'r',encoding='utf-8') as fp:
            lines=fp.readlines()
        print(len(lines),len(predict_list),len(label_list))
        emppath = ''
        if self.is_noised == 'noised':
            temppath = 'noised_'
        with open(resultpath+'txt/'+self.datatype+'_'+temppath+self.old_or_new+"_epoch"+str(epoch)+'_report_file.txt', 'w', encoding='utf-8') as fp:
            fp.write('data label read_label predict\n')
            line_num=0
            for i in range(len(label_list)):
                if lines[i]!='\n':
                    fp.write(lines[i][:-1]+' '+label_list[i-line_num]+' '+predict_list[i-line_num]+'\n')
                else:
                    fp.write('\n')
                    line_num+=1

    def get_f1(self,predict_list,label_list):
        a=0 #实体预测正确
        b=0 #实体预测错
        c=0 #O预测正确
        d=0 #O预测错
        for i in range(len(label_list)):
            if label_list[i]!='O':
                if predict_list[i]==label_list[i]:
                    a+=1
                else:b+=1
            else:
                if predict_list[i]==label_list[i]:
                    c+=1
                else:d+=1
        return a,b,c,d