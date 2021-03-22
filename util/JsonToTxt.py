import json
import numpy as np
import os
import random
import pandas as pd
from sklearn import model_selection

#生成size个取值范围在[start,end)的数字字符串
def generateNum(start,end,size):
    result = ''
    for i in range(size):
        result += str(np.random.randint(start, end))
    return result

#生成QQ
def generateQQ():
    return generateNum(1,3,1)+generateNum(0,10,9)
#生成Email
def generateEmail():
    email=['@qq.com','@163.com','@139.com','@edu.cn','@oudlook.com','@mail.com','@Gmail.com','@inbox.com']
    return generateNum(1,10,1)+generateNum(0,10,9)+email[np.random.randint(0,8)]

#生成手机号
def generatePhoneNum():
    return '1'+generateNum(0,10,10)

#返回对应的列表
def returnType(type):
    result=[]
    if type==0:
        temp='E-mail:'
        for i in range(len(temp)):
            result.append(temp[i]+' O')
    elif type==1:
        temp='QQ:'
        for i in range(len(temp)):
            result.append(temp[i]+' O')
    elif type==2:
        temp='手机:'
        for i in range(len(temp)):
            result.append(temp[i]+' O')
    return result


#生成数字实体文本及其标签
def generateNumCategoryText(noise_num):
    result=[]
    temp=['如有需要请联系我们:','如果以上内容侵犯了您的合法权益，请联系我们，我们将会尽快处理。','商务合作，业务办理请联系我们：']
    type=np.random.randint(0,3,1)
    if type==0:#email+手机
        for i in range(len(temp[0])):
            result.append(temp[0][i]+' O')

        text=generateEmail()
        result.extend(returnType(0))
        templist=[]
        templist.append(text[0]+' B-email')
        for i in range(1,len(text)-1):
            templist.append(text[i]+' M-email')
        templist.append(text[len(text)-1]+' E-email')

        for i in range(noise_num):
            noise_data=random.choice('qazmhky1234567890*-+/.')
            templist.insert(np.random.randint(2,len(templist)-2),noise_data+' H')

        result.extend(templist)
        result.append('。 O')


        text = generatePhoneNum()
        result.extend(returnType(2))
        templist = []
        templist.append(text[0] + ' B-mobile')
        for i in range(1, len(text) - 1):
            templist.append(text[i] + ' M-mobile')
        templist.append(text[len(text) - 1] + ' E-mobile')

        for i in range(noise_num):
            noise_data = random.choice('qazmhky1234567890*-+/.')
            templist.insert(np.random.randint(2, len(templist) - 2), noise_data + ' H')

        result.extend(templist)
        result.append('。 O')

    elif type==1:#QQ+手机
        for i in range(len(temp[1])):
            result.append(temp[1][i] + ' O')

        text = generateQQ()
        result.extend(returnType(1))
        templist = []
        templist.append(text[0] + ' B-QQ')
        for i in range(1, len(text) - 1):
            templist.append(text[i] + ' M-QQ')
        templist.append(text[len(text) - 1] + ' E-QQ')

        for i in range(noise_num):
            noise_data = random.choice('qazmhky1234567890*-+/.')
            templist.insert(np.random.randint(2, len(templist) - 2), noise_data + ' H')

        result.extend(templist)
        result.append('。 O')


        text = generatePhoneNum()
        result.extend(returnType(2))
        templist = []
        templist.append(text[0] + ' B-mobile')
        for i in range(1, len(text) - 1):
            templist.append(text[i] + ' M-mobile')
        templist.append(text[len(text) - 1] + ' E-mobile')

        for i in range(noise_num):
            noise_data = random.choice('qazmhky1234567890*-+/.')
            templist.insert(np.random.randint(2, len(templist) - 2), noise_data + ' H')

        result.extend(templist)
        result.append('。 O')
    elif type==2:#手机
        for i in range(len(temp[1])):
            result.append(temp[1][i] + ' O')

        text = generatePhoneNum()
        result.extend(returnType(2))
        templist = []
        templist.append(text[0] + ' B-mobile')
        for i in range(1, len(text) - 1):
            templist.append(text[i] + ' M-mobile')
        templist.append(text[len(text) - 1] + ' E-mobile')

        for i in range(noise_num):
            noise_data = random.choice('qazmhky1234567890*-+/.')
            templist.insert(np.random.randint(2, len(templist) - 2), noise_data + ' H')

        result.extend(templist)
        result.append('。 O')
    return result

#处理json文件
def write_txt(resource,indexs,outputpath,train_or_test,noise_num,num_percent):
    current_index=0
    num=int(len(resource)*num_percent)
    with open(outputpath[:-4] + '_'+train_or_test+'.txt', "a", encoding="utf-8") as txt:
        for index in indexs:  # 每个文本
            json_data = json.loads(resource[index])
            content = json_data["text"].replace("\n", "&").replace("\r", "&")
            data = list(content)
            myMark = ["O" for x in range(0, len(data))]
            tempDic1 = json_data["label"]
            for key1, value1, in tempDic1.items():  # 这里的key是类别，value是字典（"实体"，下标）
                for key2, value2 in value1.items():  # 这里key是内容，value是下标，二维数组
                    for i in range(len(value2)):
                        myMark[value2[i][0]] = "B-" + key1
                        for index in range(value2[i][0] + 1, value2[i][1]):
                            myMark[index] = "M-" + key1
                        myMark[value2[i][1]] = "E-" + key1

            for i in range(0, len(data)):
                txt.write(str(data[i]) + " " + str(myMark[i]) + "\n")
            if current_index<num:
                numbers = generateNumCategoryText(noise_num)
                for i in range(len(numbers)):
                    txt.write(numbers[i] + '\n')
            current_index+=1
            txt.write("\n")

#把json文件处理成两列的txt文件
def transformJSONFile(inputPath,outputPath,noise_num,split_size,num_percent):#转换JSON文件成训练集
    with open(inputPath, 'r', encoding='utf-8') as fp:
        lines=fp.readlines()
    index=[x for x in range(len(lines))]
    if split_size==0:
        write_txt(lines,index,outputPath,'train',noise_num,num_percent)
    else:
        train, test = model_selection.train_test_split(index, test_size=split_size, random_state=1)
        write_txt(lines,train,outputPath,'train',noise_num,num_percent)
        write_txt(lines, test, outputPath, 'test',noise_num,num_percent)

#处理CCF数据集，转化为json文件
def del_CCF_Data():
    file=r'../resource/ccf_14.json'
    CCF_file = r"../resource/"
    with open(file, 'w', encoding='utf-8', newline='\n') as json_file:
        for i in range(2515):
            result = {}
            # 读取文本放入data列表
            with open(CCF_file + 'data/' + str(i) + '.txt', "r", encoding="utf-8") as fp:
                text = fp.read()
            text = text.replace("\n", "&").replace("\r", "&")
            result['text'] = text

            # 读取label并根据label的内容进行数据处理并放入myMark
            csv_file = pd.read_csv(CCF_file + 'label/' + str(i) + '.csv', sep=',')
            label = {}
            for row in csv_file.iterrows():
                if row[1]['Category'] not in label.keys():  # 类别不存在
                    label[row[1]['Category']] = {}
                if row[1]['Privacy'] not in label[row[1]['Category']].keys():  # 实体不存在
                    label[row[1]['Category']][row[1]['Privacy']] = []
                label[row[1]['Category']][row[1]['Privacy']].append([row[1]['Pos_b'], row[1]['Pos_e']])
            result['label'] = label

            # 写文件
            json.dump(result, json_file, ensure_ascii=False)
            json_file.write('\n')

#处理微博数据集，转化为txt文件
def del_Weibo_Data(inputfilepath,outputfile,noise_num,num_percent):
    file=r"../resource/WeiboNER/"
    with open(inputfilepath, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    file_num=0
    for line in lines:
        if line=='\n':
            file_num+=1
    num=int(file_num*num_percent)
    current_index=0
    with open(outputfile, 'w', encoding='utf-8') as rs:
        for line in lines:
            if line=='\n':
                if current_index<num:
                    numbers = generateNumCategoryText(noise_num)
                    for i in range(len(numbers)):
                        rs.write(numbers[i] + '\n')
                current_index+=1
            elif len(line)>4:
                line=line[:-5]+'\n'
            rs.write(line)

file=r'../resource/'

transformJSONFile(file+'ccf_14.json',file+'old/ccf_14_noised.txt', 2, 0.2,0.2)

transformJSONFile(file+'cluener_10.json',file+'old/cluener_10_noised.txt', 2,0,0.2)
transformJSONFile(file+'cluener_10_test.json',file+'old/cluener_10_noised.txt', 2,0,0.2)

del_Weibo_Data(r'../resource/WeiboNER/train.all.bmes',r'../resource/old/weibo_4_noised_train.txt',2,0.2)
del_Weibo_Data(r'../resource/WeiboNER/test.all.bmes',r'../resource/old/weibo_4_noised_test.txt',2,0.2)