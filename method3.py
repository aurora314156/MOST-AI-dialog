
# -*- coding: UTF-8 -*-
import time
import jieba
import os
from gensim.models import word2vec
from gensim import models
import numpy as np
from scipy import spatial


# 精確模式 ：將句子最精確地切開，叫適合文本分析, cut_all=False
# 全模式：把句子中所有的可以成詞的詞語都掃描出來, 速度快, cut_all=True
# 搜索引擎模式：在精確模式的基礎上對長詞再次切分，提高召回率，適合用於搜尋引擎分詞, jieba.cut_for_search(Content)            
# call jieba api
def jiebaCut(s):
    #words = jieba.cut(s,cut_all=True)
    words = jieba.cut_for_search(s)
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



def main():
    sTime = time.time()
    result = 'gigaword_sum_search_result.txt'
    print("Start process CQA dataset")
    cNum, accuracy = 0, 0
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
                ans = s
                print("Corpus: %d" % cNum)
                guessAns = word2VecSum(cList, qList, aList, cNum)
                #guessAns = word2VecAve(cList, qList, aList, cNum)
                with open(result, 'a') as res:
                    res.write("\nCorpus :" + str(cNum))
                    res.write("\nCorrect answer is: " + ans)
                    res.write("\nPredict answer is: " + guessAns)
                    res.write('\n')
                if guessAns == ans:
                    accuracy +=1
                print("====== Final result ======")
                print("Correct answer is: %s." %(ans))
                print("Predict answer is: %s.\n" %(guessAns))
                #print("corpus:\n",cList,'\nquestion:\n',qList,'\nanswer:\n',aList,'\ncorrect ans:\n',ans,'\n')
                cList, qList, aList = [],[],[]
                flag, end = 0, 0
                ans = ""
                cNum +=1
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
    
    with open(result, 'a') as res:
        res.write("\nTotal corpus number :" + str(cNum))
        res.write("\nAccuracy is :" + str(accuracy/cNum*100))
    print("\nTotal corpus numbers: %d" % cNum)
    print("Accuracy is %.3f percent" % (accuracy/cNum*100))
    print("Processing all CQA dataset corpus took %.2fs" % (time.time()- sTime))
        


def word2VecAve(cList, qList, aList, cNum):

    sTime = time.time()
    print("====== Start process words vector sum ======")
    nc = np.zeros((len(cList),250),dtype=float)
    nq = np.zeros(250,dtype=float)
    na = np.zeros((len(aList),250),dtype=float)
    count, ind, notExist = 0 , 0 , 0
    # take all element from corpus List
    for c in cList:
        for w in c:
            # take word vector from word2vec model
            try:
                m = model[w]
            except KeyError as e:
                notExist +=1
                continue
            # calculate word vector sum from corpus list
            for n in range(250):
                nc[ind][n] += m[n]
            count +=1
        for m in range(250):
            nc[ind][m] /= len(c)
        ind +=1
    # take all element from question List
    for w in qList:
        try:
            m = model[w]
        except KeyError as e:
            notExist +=1
            continue
        # calculate word vector sum from question list
        for n in range(250):
            nq[n] += m[n]
        count +=1
    for m in range(250):
        nq[m] /= len(qList)
        
    
    ind = 0
     # take all element from answer List
    for a in aList:
        for w in a:
            try:
                m = model[w]
            except KeyError as e:
                notExist +=1
                continue
             # calculate word vector sum from answer list
            for n in range(250):
                na[ind][n] += m[n]
            count +=1
        for m in range(250):
            na[ind][m] /= len(aList)
        ind +=1
        
    print("This corpus has total %d split words." % (count))
    print("This corpus has %d words not in word2vec model." % (notExist))
    print("Process all corpus content took %.2fs." % (time.time()- sTime))
    # go to final step, calculate similarity
    guessAns = similarity(nc, nq, na, cNum)
    return guessAns
    


def word2VecSum(cList, qList, aList, cNum):

    sTime = time.time()
    print("====== Start process words vector sum ======")
    nc = np.zeros((len(cList),250),dtype=float)
    nq = np.zeros(250,dtype=float)
    na = np.zeros((len(aList),250),dtype=float)
    count, ind, notExist = 0 , 0 , 0
    # take all element from corpus List
    for c in cList:
        for w in c:
            # take word vector from word2vec model
            try:
                m = model[w]
            except KeyError as e:
                notExist +=1
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
            notExist +=1
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
                notExist +=1
                continue
             # calculate word vector sum from answer list
            for n in range(250):
                na[ind][n] += m[n]
            count +=1
        ind +=1
        
    print("This corpus has total %d split words." % (count))
    print("This corpus has %d words not in word2vec model." % (notExist))
    print("Process all corpus content took %.2fs." % (time.time()- sTime))
    # go to final step, calculate similarity
    guessAns = similarity(nc, nq, na, cNum)
    return guessAns
    


def similarity(nc, nq, na, cNum):
    
    sTime = time.time()
    l = ['A','B','C','D']
    print("====== Start process vector similarity ======")
    # highest corpus/answer similarity
    h_c_Sim, h_a_Sim, highCorpus, ans = 0, 0, 0, 0
    
    # calculate the most similar corpus and question
    for c in nc:
        cosSim = 1 - spatial.distance.cosine(c, nq)
        if cosSim > h_c_Sim:
            h_c_Sim = cosSim
            # record highest similarity corpus
            highCorpus = c
            
    # calculate the most similar corpus and answer
    i = 0
    for a in na:
        cosSim = 1 - spatial.distance.cosine(a, highCorpus)
        if cosSim > h_a_Sim:
            h_a_Sim = cosSim
            ans = i
        i += 1
    
    print("The best match answer to this CQA is %s." %(l[ans]))
    print("The best match answer similarity to this CQA is %.2f." %(h_a_Sim))
    print("Process all similarity calculation took %.2fs.\n" % (time.time()- sTime))
    return l[ans]



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

