# -*- coding: UTF-8 -*-
import time
import json
import os
import shutil
import re

# read cutted data
def processFile(line, ind):
    result, ansList, temp = [], [], []
    state, check = 0, 1
    for l in line:
        tmp = []
        l = l.strip()

        for i in range(len(l)):
            index = l.find("＜")
            if index == -1:
                continue
            l = l[index+8:]

        if len(l) == 1 and state < 3:
            state += 1
            continue

        # Process C and Q
        if state < 3:
            content = re.sub('。',' ', l)
            s = content.split(" ")
            result.append(s)

        # Process A
        else:
            l = l[1:]
            content = re.sub('。',' ', l)
            if content == '':
                continue
            ansList.append(content.split(" "))

    if len(ansList) < 4:
        result = []
        for t in range(3):
            result.append(temp)
        outputFile(result, ind)
        check = 0
        return check

    result.append(ansList)
    # output result to json
    outputFile(result, ind)
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
    for f in range(7000):
        if os.path.exists(filePath + str(f) + '.txt'):
            with open(filePath + str(f) + '.txt', 'r') as file:
                print(f)
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

# ====== initial setting ======


if __name__ == "__main__":
    main()
