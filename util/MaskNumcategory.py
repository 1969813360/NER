oldfilepath='../resource/old/'
newfilepath='../resource/new/'
def mask(filename,mask_lenth):
    with open(oldfilepath+filename,'r',encoding='utf-8') as fp:
        lines=fp.readlines()
    with open(newfilepath+filename,'a',encoding='utf-8') as writer:
        num_length=0
        temp=[]
        for line in lines:
            if str.isdigit(line[0]):
                 temp.append(line)
                 num_length+=1
            else:
                if num_length>=mask_lenth:#对连续的 >=mask_lenth个数字用两个*字符mask掉
                    if temp[0][1:]==temp[-1][1:]:
                        writer.write('*'+temp[0][1:]+line)
                    else:
                        writer.write('*'+temp[0][1:]+'*'+temp[-1][1:]+line)
                elif num_length>0 and num_length<mask_lenth:
                    for i in range(len(temp)):
                        writer.write(temp[i])
                    writer.write(line)
                else:
                    writer.write(line)
                temp=[]
                num_length=0

#*************************************************
mask('ccf_14_noised_train.txt',4)
mask('ccf_14_noised_test.txt',4)

mask('cluener_10_noised_train.txt',4)
mask('cluener_10_noised_test.txt',4)

mask('weibo_4_noised_train.txt',4)
mask('weibo_4_noised_test.txt',4)