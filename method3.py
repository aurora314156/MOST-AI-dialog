
# coding: utf-8

# In[25]:


# -*- coding: UTF-8 -*-
import time
import jieba
from gensim.models import word2vec
from gensim import models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial


# In[26]:


# 精確模式 ：將句子最精確地切開，叫適合文本分析, cut_all=False
# 全模式：把句子中所有的可以成詞的詞語都掃描出來, 速度快, cut_all=True
# 搜索引擎模式：在精確模式的基礎上對長詞再次切分，提高召回率，適合用於搜尋引擎分詞, jieba.cut_for_search(Content)            
# call jieba api
def jiebaCut(s):
    words = jieba.cut(s, cut_all=False)
    result = removeStopWords(words)
    return result

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


# In[27]:


def main():
    sTime = time.time()
    print("Start process CQA dataset")
    cNum = 0
    with open('CQA.txt', 'r') as file:
        flag, end = 0, 0
        cList, qList, aList = [],[],[]
        tempC = []
        ans = ""
        for i in file.readlines():
            s = i.strip()
            flag, nextline = state(s,flag)
            # one corpus process done!
            if end == 4:
                cNum +=1
                ans = s
                word2VecSum(cList, qList, aList, cNum)
                print("Corpus: %d" % cNum)
                #print("corpus:\n",cList,'\nquestion:\n',qList,'\nanswer:\n',aList,'\ncorrect ans:\n',ans,'\n')
                cList, qList, aList = [],[],[]
                flag, end = 0, 0
                ans = ""
                continue
            # still on state
            if nextline != 1:
                continue
            # on state 1, process Corpus
            elif flag == 1:
                cutRes = jiebaCut(s)
                for c in cutRes:
                    tempC.append(c)
                if nextline == 1:
                    if tempC:
                        cList.append(tempC)
                        tempC = []
            # on state 2, process Question
            elif flag == 2:
                cutRes = jiebaCut(s)
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
                cutRes = jiebaCut(tempS)
                tempL = []
                for c in cutRes:
                    tempL.append(c)
                aList.append(tempL)

        print("\nTotal corpus numbers: %d" % cNum)
        print("Processing all CQA dataset corpus took %.2fs" % (time.time()- sTime))


# In[28]:



def word2VecSum(cList, qList, aList, cNum):

    sTime = time.time()
    print("Start process words vector sum, corpus : %d" % (cNum))
    nc = np.zeros((len(cList),250),dtype=float)
    nq = np.zeros(250,dtype=float)
    na = np.zeros((len(aList),250),dtype=float)
    count, ind = 0 , 0
    # take all element from corpus List
    for c in cList:
        for w in c:
            # take word vector from word2vec model
            print(w)
            try:
                m = model[w]
            except KeyError as e:
                print('this word is not in model.')
                continue
            # calculate word vector sum from corpus list
            for n in range(250):
                nc[ind][n] += m[n]
            count +=1
        ind +=1
    # take all element from question List
    for w in qList:
        try:
            m = model[w]
        except KeyError as e:
            print('this word is not in model.')
            continue
        # calculate word vector sum from question list
        for n in range(250):
            nq[n] += m[n]
        count +=1
        
    ind = 0
     # take all element from answer List
    for a in aList:
        for w in a:
            try:
                m = model[w]
            except KeyError as e:
                print('this word is not in model.')
                continue
             # calculate word vector sum from answer list
            for n in range(250):
                na[ind][n] += m[n]
            count +=1
        ind +=1
        
    print("This corpus has total %d split words." % (count))
    print("Process all corpus content took %.2fs.\n" % (time.time()- sTime))
    # go to final step, calculate similarity
    #similarity(nc, nq, na, cNum)
    
    


# In[ ]:


def similarity(nc, nq, na, cNum):
    sTime = time.time()
    print("Start process vector similarity, corpus : %d" % (cNum))
    print(nc)
    print(nq)
    for q in nq:
        for c in nc:
            highSim = 0
            #cosSim = cosine_similarity(c,q)
            cosSim =  1 - spatial.distance.cosine(c, q)
            if cosSim > highSim:
                highSim = cosSim
                highC = c
        print(highSim)
    
    print("Process all similarity calculation took %.2fs.\n" % (time.time()- sTime))


# In[ ]:


# ====== initial setting ======

print("Start loading initial setting!")
# jieba setting
relativePath = os.getcwd()
jieba.set_dictionary(relativePath + '/jieba_setting/dict.txt.big')
# add user dictionary to improve jieba cut precision
# jieba.load_userdict(relativePath + '/jieba_setting/yourfile.txt')

# stopwords setting
stopWordsSet = set()
with open(relativePath + '/jieba_setting/stopwords.txt', 'r') as stop:
    for i in stop:
        stopWordsSet.add(i.strip('\n'))

# load word2vec model
sTime = time.time()
model = models.Word2Vec.load(relativePath + '/wiki/python/word2vec.model')
print("Load word2vec model success! took %.2fs" % (time.time()-sTime))

# ====== initial setting ======

if __name__ == "__main__":
    main()

