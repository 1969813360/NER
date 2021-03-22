import pandas as pd
path=r'./result/txt/'
def getEntityAcc(file,type):
    data = pd.read_csv(path+file + '_noised_'+type+'_report_file.txt', sep=' ')
    right=0
    wrong=0
    total=0
    row=0
    # print(data.index)
    # print(data.iloc[85379][0])
    try:
        while row<len(data.index):
            if data.iloc[row][1][0]=='B':
                total+=1
                length=1
                while row+length<len(data.index) and data.iloc[row+length][1][0]!='E':
                    # print(row+length)
                    length+=1
                label=''
                predict=''
                for i in range(length+1):
                    label+=data.iloc[row+i][1][0]
                    predict+=data.iloc[row+i][3][0]
                if label==predict:
                    right+=1
                else:
                    wrong+=1
                row=row+length+1
            else:
                row+=1
    except:
        print('row:',row,'length:',length)
    print('total:',total,'r:',right,'w:',wrong)

getEntityAcc('cluener','old')