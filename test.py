from sklearn import model_selection,metrics
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as pl
import jieba

# with open('./resource/cluener_10.json','r',encoding='utf-8') as fp:
#     lines=fp.readlines()
#
lenth=[]
for line in lines:
    json_data = json.loads(line)
    lenth.append(len(json_data['text']))
fig = pl.figure(figsize=(6, 6))
pl.hist(lenth)
fig.savefig('D:/temp/cluener.jpg',dpi=600,bbox_inches='tight')
pl.show()


# with open(r'./resource/WeiboNER/train.all.bmes', 'r',encoding='utf-8') as fp:
#     lines=fp.readlines()
# lens=[]
# len=0
# for line in lines:
#     if line!='\n':
#         len+=1
#     else:
#         lens.append(len)
#         len=0
# fig = pl.figure(figsize=(6, 6))
# pl.hist(lens)
# fig.savefig('D:/temp/weibo.jpg',dpi=600,bbox_inches='tight')
# pl.show()


# a='民主革命积极分子孙中山上书李鸿章。'
# result=jieba.lcut(a,cut_all=True)
# print(result)

# data=pd.read_csv(r'./resource/new/weibo_4_noised_test.txt', sep=' ',header=None,skiprows=[71758])
# print(data.groupby(1).count())

print(int(101*0.2))