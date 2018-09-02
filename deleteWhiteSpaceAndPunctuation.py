# -*- coding: UTF-8 -*-
import time
import json
import os
import shutil
import re



# read cutted data
def processFile(line, ind):
    for l in line:
        l = l.strip()
        l = l.replace(' ', '')

        line = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+","",l)
        with open("transfer-out/" +str(ind)+ '.txt', 'a') as f:
            f.writelines(line+'\n')


# main func start point
def main():
    sTime = time.time()
    print("Start delete CQA whitespace")
    ind, wrongtotal = 0, 0
    filePath = os.getcwd() + '/preprocess/'
    # start process CQA data set
    for f in range(7000):
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

# ====== initial setting ======


if __name__ == "__main__":
    main()
