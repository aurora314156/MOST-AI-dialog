
# coding: utf-8

# In[9]:


# -*- coding: UTF-8 -*-
import time
import jieba


# In[10]:


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


# In[11]:


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


# In[12]:



sTime = time.time()
print("Start process CQA dataset")
cNum = 0
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
            print("Corpus: %d" % cNum)
            print("corpus:\n",cList,'\nquestion:\n',qList,'\nanswer:\n',aList,'\ncorrect ans:\n',ans,'\n')
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
                cList.append(c)
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

