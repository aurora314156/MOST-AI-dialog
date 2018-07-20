# -*- coding: UTF-8 -*-
import time
import json
import os
import shutil
import re
import jieba



# call jieba api
def jiebaCut(s):
    # 精確模式 ：將句子最精確地切開，叫適合文本分析, cut_all=False
    # 全模式：把句子中所有的可以成詞的詞語都掃描出來, 速度快, cut_all=True
    # 搜索引擎模式：在精確模式的基礎上對長詞再次切分，提高召回率，適合用於搜尋引擎分詞, jieba.cut_for_search(Content)
    words = jieba.cut_for_search(s)
    return words


# process whole file
def processFile(line, ind):
    result,ansList, temp = [], [], []
    state, check = 0, 1
    for l in line:
        temp = []
        l = l.strip()
        if len(l) == 1 and state < 3:
            state += 1
            continue
        # process C and Q, without jieba cut
        if state < 3:
            content = re.sub('。',' ',l)
            for c in content.split(" "):
                temp.append(c)
            result.append(temp)
        # process A, use jieba cut
        else:
            if len(l) == 0:
            # if answers are not correcct format
                result = []
                for t in range(3):
                    result.append(temp)
                outputFile(result, ind)
                check = 0
                return check
            # if answers are correct format
            # 如果答案有一二三四開頭，先移除它們
         
            if len(l) > 2:
                l = l[1:]
                cutRes = jiebaCut(l)
                for c in cutRes:
                    temp.append(c)
            # 如果答案只有一個字，沒辦法斷詞的時候
            else:
                temp.append(l)
            ansList.append(temp)
    result.append(ansList)
    # output result to json
    outputFile(result,ind)
    return check
    

# output result file
def outputFile(result, ind):
    content = {}
    fileName = 'Result/' + '/' + str(ind) + '.json'
    with open(fileName, 'w') as CQA:
        content["corpus"] = result[0]
        content["question"] = result[1]
        content["answer"] = result[2]
        CQA.write(json.dumps(content, ensure_ascii=False))

# main func start point
def main():
    sTime = time.time()
    print("Start process CQA dataset")
    ind, wrongtotal = 0, 0
    filePath = os.getcwd() + '/CQA/'
    # start process CQA data set
    for f in range(30000):
        if os.path.exists(filePath + str(f) + '.txt'):
            with open(filePath + str(f) + '.txt', 'r') as file:
                check = processFile(file.readlines(), ind)
                if check == 0:
                    #print("\nCorpus : %d have wrong answer format." % f)
                    wrongtotal += 1
                ind += 1

    print("\nTotal corpus numbers: %d" % ind)
    print("\nWrong answer of corpus numbers: %d" % wrongtotal)
    print("Processing all CQA dataset corpus took %.2fs" % (time.time()- sTime))


# ====== initial setting ======
print("Start loading initial setting!")
# jieba setting
print("Start loading jieba dictionary!")
relativePath = os.getcwd()
jieba.set_dictionary(relativePath + '/jieba_setting/dict.txt.big')


# ====== initial setting ======


if __name__ == "__main__":
    main()
