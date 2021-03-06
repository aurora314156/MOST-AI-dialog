# -*- coding: UTF-8 -*-
import time
import json
import os
import shutil
import re
from os import listdir


def strQ2B(ustring):

    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288: # 全形空格直接轉換
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374): # 全形字元（除空格）根據關係轉化
                inside_code -= 65248
            rstring += chr(inside_code)
            ss.append(rstring)
    return ss


# return list of cut word
def cut_list(l):
    temp = []
    for ll in l.split(" "):
        if ll is not " ":
            temp.append(ll)
    return temp



# read cutted data
def processFile(line, fileName):
    flag = 0
    i = 0
    result = []
    temp = []
    for l in line:
        l = l.strip()
        cl = ""
        if flag == 0 or flag == 2 or flag == 4:
            flag +=1
            continue
        else:
            cl = cut_list(l)
        if cl is not "" and flag < 5:
            result.append(cl)
        elif flag>4 and flag <9:
            temp.append(cl)
        elif flag == 9:
            result.append(temp)
            result.append(strQ2B(l[:1]))
        flag += 1
    
    outputFile(result, fileName)
        

# output result file
def outputFile(result, fileName):
    content = {}
    fileName = 'JsonResultFile/' + '/' + str(fileName) + '.json'
    with open(fileName, 'w') as CQA:
        content["corpus"] = result[0]
        content["question"] = result[1]
        content["answer"] = result[2]
        content["correct_answer"] = result[3]
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
CQAPath = os.getcwd() + '/OriginFile/'
files = listdir(CQAPath)
print("Start convert txt to json")

# ====== initial setting ======


if __name__ == "__main__":
    main()
