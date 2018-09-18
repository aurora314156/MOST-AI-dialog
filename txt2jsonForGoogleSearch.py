# -*- coding: UTF-8 -*-
import time
import json
import os
import shutil
import re
from os import listdir


# read cutted data
def processFile(line, fileName):
    result = []
    error = {'＜','ＵＮＫ＞', '\n', ''}
    for l in line:
        else:
            for word in l.split(" "):
                if word not in error:
                    result.append(word)

    outputFile(result, fileName)
        

# output result file
def outputFile(result, fileName):
    content = {}
    fileName = 'JsonResultFile/' + '/' + str(fileName) + '.json'
    with open(fileName, 'w') as CQA:
        content["corpus"] = result
        CQA.write(json.dumps(content, ensure_ascii=False))

# main func start point
def main():
    ind = 0
    sTime = time.time()
    print("Start process CQA dataset")
    # start process CQA data set
    for ff in files:
        with open(CQAPath + ff, 'r') as file:
            processFile(file.readlines(), ff.strip('.txt'))
            ind += 1

    print("\nTotal corpus numbers: %d" % ind)
    print("Processing all CQA dataset corpus took %.2fs" % (time.time()- sTime))

# ====== initial setting ======
CQAPath = os.getcwd() + '/GoogleSearch/'
files = listdir(CQAPath)
print("Start convert txt to json")

# ====== initial setting ======


if __name__ == "__main__":
    main()
