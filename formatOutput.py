# -*- coding: UTF-8 -*-
import time
import jieba
import json
import os
from gensim import models
import shutil


# remove stopwords
def removeStopWords(words):
    result = []
    for w in words:
        if w not in stopWordsSet:
            result.append(w)
    return result

# define all state 
def state(s,flag):
    nextline = 1
    if s is None or s == "":
        return flag, nextline
    # state: 1, s[0] = C
    if s[0] == 'C':
        flag, nextline = 1, 0
    # state: 2, s[0] = Q
    elif s[0] == 'Q':
        flag, nextline = 2, 0
    # state: 3, s[0] = A
    elif s[0] == 'A':
        flag, nextline = 3, 0
    # state: 4, do jieba cut
    return flag, nextline

# call jieba api
def jiebaCut(s, m):
    # 精確模式 ：將句子最精確地切開，叫適合文本分析, cut_all=False
    # 全模式：把句子中所有的可以成詞的詞語都掃描出來, 速度快, cut_all=True
    # 搜索引擎模式：在精確模式的基礎上對長詞再次切分，提高召回率，適合用於搜尋引擎分詞, jieba.cut_for_search(Content)  
    if m == 0:
        words = jieba.cut(s, cut_all=False)
    elif m == 1:
        words = jieba.cut(s, cut_all=True)
    elif m == 2:
        words = jieba.cut_for_search(s)

    result = removeStopWords(words)
    return result

# output result file
def outputFile(cList, qList, aList, ans, ind, m):
    content = {}
    res, name = [], ['cut_all_false', 'cut_all_true', 'cut_for_search']
    fileName = 'CQA/'+ name[m] + '/CQA_' + str(ind) + '.json'
    with open(fileName, 'w') as CQA:
        content["corpus"] = cList
        content["question"] = qList
        content["answer"] = aList
        content["correct_answer"] = ans
        CQA.write(json.dumps(content, ensure_ascii=False))

# main func start point
def main():
    sTime = time.time()
    print("Start process CQA dataset")
    for m in range(3):
        cNum ,ind = 0, 0
        # remove all file under the folder
        name = ['cut_all_false', 'cut_all_true', 'cut_for_search']
        shutil.rmtree('CQA/'+name[m])  
        os.mkdir('CQA/'+name[m])
        # start process CQA data set
        with open('CQA.txt', 'r') as file:
            flag, end = 0, 0
            cList, qList, aList = [],[],[]
            ans = ""
            for i in file.readlines():
                s = i.strip()
                flag, nextline = state(s,flag)
                # one corpus process done!
                if end == 4:
                    cNum +=1
                    ans = s
                    outputFile(cList, qList, aList, ans, ind, m)
                    cList, qList, aList = [],[],[]
                    flag, end = 0, 0
                    ans = ""
                    ind += 1
                    continue
                # still on state
                if nextline != 1:
                    continue
                # on state 1, process Corpus
                elif flag == 1:
                    cutRes = jiebaCut(s, m)
                    for c in cutRes:
                        cList.append(c)
                # on state 2, process Question
                elif flag == 2:
                    cutRes = jiebaCut(s, m)
                    for c in cutRes:
                        qList.append(c)
                # on state 3, process Answer
                elif flag == 3:
                    end += 1
                    # example: （B） 吃飯比讀書更為重要 
                    tempS = ""
                    skip = ['A','B','C','D','（',')']
                    check = 0
                    for j in s:
                        if check == 3:
                            tempS += j
                        else:
                            check += 1
                    tempS = tempS.strip()
                    cutRes = jiebaCut(tempS, m)
                    tempL = []
                    for c in cutRes:
                        tempL.append(c)
                    aList.append(tempL)
                    
        print("\nTotal corpus numbers: %d" % cNum)
        print("Processing all CQA dataset corpus took %.2fs" % (time.time()- sTime))


# ====== initial setting ======

print("Start loading initial setting!")
# jieba setting
print("Start loading jieba dictionary!")
relativePath = os.getcwd()
jieba.set_dictionary(relativePath + '/jieba_setting/dict.txt.big')
# add user dictionary to improve jieba cut precision
# jieba.load_userdict(relativePath + '/jieba_setting/yourfile.txt')

# stopwords setting
print("Start add stopwords!")
stopWordsSet = set()
with open(relativePath + '/jieba_setting/stopwords.txt', 'r') as stop:
    for i in stop:
        stopWordsSet.add(i.strip('\n'))

# load word2vec model
print("Start loading word2vec model!")
sTime = time.time()
model = models.Word2Vec.load(relativePath + '/gigaword/python/word2vec.model')
print("Load word2vec model success! took %.2fs" % (time.time()-sTime))

# ====== initial setting ======


if __name__ == "__main__":
    main()
