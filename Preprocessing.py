


import sys
import os
import time
import re
from os import listdir

originFile = os.getcwd() + '/OriginFile/'
resultFolder = os.getcwd() +'/PreResultFile/'
files = listdir(originFile)


def reserveLang(char):
    appendWord = ""
    # reserve zh
    if '\u4e00' <= char <= '\u9fff' or '\u2e80'<= char <= '\u2fdf' or '\u3400'<= char <= '\u4dbf':
       appendWord = char
    # reserve number
    elif char >= u'\u0030' and char <= u'\u0039':
       appendWord = char
    return appendWord

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
                    word = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）「」；:：；]+","",l)
                    for char in word.encode('utf-8', 'ignore').decode('utf-8'):
                        # reserve option function
                        #result += reserveLang(char)
                        # reserve all
                        result += char
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
