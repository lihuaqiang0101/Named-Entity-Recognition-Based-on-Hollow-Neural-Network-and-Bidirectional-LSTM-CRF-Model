#encoding=utf8
import os,jieba,csv
import jieba.posseg as pseg#导入词性标注
c_root=os.getcwd()+os.sep+"source_data"+os.sep#os.getcwd()是当前文件所在的路径，os.sep是\
dev=open("example.dev",'w',encoding='utf8')
train=open("example.train",'w',encoding='utf8')
test=open("example.test",'w',encoding='utf8')
#对词语所要打的标记
biaoji = set(['DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT', 'Dur', 'TP', 'REG', 'ORG', 'AT', 'PSB', 'DEG', 'FW','CL'])
fuhao=set(['。','?','？','!','！'])
dics=csv.reader(open("DICT_NOW.csv",'r',encoding='utf8'))#症状所对应的标记的字典
# 将医学专有名词以及标签加入结巴词典中
for row in dics:
    if len(row)==2:#排除没有打标记的关键词，保留打了标记的
        jieba.add_word(row[0].strip(),tag=row[1].strip())            # 保证添加的词语不会被cut掉
        jieba.suggest_freq(row[0].strip())                           # 可调节单个词语的词频，使其能（或不能）被分出来。
split_num=0
for file in os.listdir(c_root):#列出每个病人的病历
    if "txtoriginal.txt" in file:#判断是否是身体状况
        fp=open(c_root+file,'r',encoding='utf8')#打开每一个这样的文件
        for line in fp:#遍历每一个文件中每一行
            split_num+=1#每遍历一次就记一次数，主要作用是将数据进行分块，2/5用于验证2/5用于测试1/5用于训练
            words=pseg.cut(line)#对每一行进行分词然后对每一个词进行词性标注返回一个这行的迭代器，现在cut出来的就包含了之前添加进词典的
            for key,value in words:
                #print(key)
                # print(value)
                if value.strip() and key.strip():#保证关键字和词性都不是空的
                    import time
                    start_time=time.time()
                    #split_num%15<2（0，1）：1，split_num%15>1 and split_num%15<4（2，3）：2，split_num%15=4：3
                    index=str(1) if split_num%15<2 else str(2)  if split_num%15>1 and split_num%15<4 else str(3)
                    end_time=time.time()
                    print("method one used time is {}".format(end_time-start_time))
                    if value not in biaoji:#如果改词性标注没有在biaoji里面就同意标记为0（即非实体）
                        value='O'
                        for achar in key.strip():#拿到每个词中的每个字
                            if achar and achar.strip() in fuhao:#如果这个字符不是空的并且是标点符号
                                string=achar+" "+value.strip()+"\n"+"\n"
                                dev.write(string) if index=='1' else test.write(string) if index=='2' else train.write(string)
                            elif achar.strip() and achar.strip() not in fuhao:
                                string = achar + " " + value.strip() + "\n"
                                dev.write(string) if index=='1' else test.write(string) if index=='2' else train.write(string)

                    elif value.strip()  in biaoji:
                        begin=0
                        for char in key.strip():
                            if begin==0:
                                begin+=1
                                string1=char+' '+'B-'+value.strip()+'\n'#初次出现就打个B-
                                if index=='1':
                                    dev.write(string1)
                                elif index=='2':
                                    test.write(string1)
                                elif index=='3':
                                    train.write(string1)
                                else:
                                    pass
                            else:
                                string1 = char + ' ' + 'I-' + value.strip() + '\n'
                                if index=='1':
                                    dev.write(string1)
                                elif index=='2':
                                    test.write(string1)
                                elif index=='3':
                                    train.write(string1)
                                else:
                                    pass
                    else:
                        continue
dev.close()
train.close()
test.close()