


import sys
import os
import time
import re
from os import listdir

originFile = os.getcwd() + '/OriginFile/'
resultFolder = os.getcwd() +'/PreResultFile/'
files = listdir(originFile)


i = 0
for ff in files:
    sTime = time.time()
    f = open(originFile + ff, 'r')
    with open(resultFolder + ff, 'w', newline='') as w:
        for line in open(originFile + ff):
            line = f.readline()
            result = ""
            for l in line:
                if l is not "\\" and l is not " ":
                    # remove punctuation
                    word = re.sub("[<UNK>\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+","",l)
                    if word is not "":
                        for char in word.encode('utf-8', 'ignore').decode('utf-8'):
                            # remove all word without zh
                            #if '\u4e00' <= char <= '\u9fff' or '\u2e80'<= char <= '\u2fdf' or '\u3400'<= char <= '\u4dbf':
                            result+=char
                    else:
                        word = " "
                        result += word
            ind = 0
            tmp = ""
            for r in result:
                ind += 1
                tmp += r
                if ind % 5000 == 0:
                    w.write(tmp)
                    w.write("\n")
                    tmp = ""
            if tmp is not "":
                w.write(tmp)
                w.write("\n")
            i += 1
            if i% 50000 ==0:
                print(i)
    print("Process File {n} took {s}s.".format(n=ff, s=time.time()-sTime))
