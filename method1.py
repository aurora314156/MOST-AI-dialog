
# coding: utf-8

from gensim.models import word2vec
from gensim import models
from pprint import pprint
from scipy import spatial
import numpy as np
import time
import os
import json
import random
import csv
from os import listdir


def readData(path):
    t = time.time()
    with open(path, 'r') as reader:
        data = json.loads(reader.read())
    return data

# ==================
#    method 1
# ==================
def generateAnswer(data, model):
    CQ_con = np.zeros(250, dtype = float)
    A_con = np.zeros((250, 250), dtype = float)
    #ca = data['correct_answer']
    anslist = ['1', '2', '3', '4']
    CQ_list = data['corpus']
    CQ_list.extend(data['question'])

    for word in CQ_list:
        try:
            vector = model[word]
        except KeyError as e:
            continue
        for i in range(250):
            CQ_con[i] += vector[i]
            
    for i in range(250):
        CQ_con[i] /= 250

    for j in range(0, 4):
        for word in data['answer'][j]:
            try:
                vector = model[word]
            except KeyError as e:
                continue
            for i in range(250):
                A_con[j][i] += vector[i]
            for i in range(250):
                A_con[j][i] /= 250
    ini = 0
    high_cq = 0
    i = 0
    ans = 0
    for a in A_con:
        cos = 1 - spatial.distance.cosine(a, CQ_con)
        if cos > ini:
            ini = cos
            high = a
            ans = i
        i += 1

    return anslist[ans]


def main():

    t = time.time()
    pathModel = './word2vec/'
    pathData = './JsonResult/'
    pm = listdir(pathModel)
    # clear result
    f = open('method1_result.txt', 'w')
    f.close()
    totalData = 1500
    correctAnsList = [4,1,3,4,3,2,3,2,1,1,3,2,3,3,3,4,2,2,3,2,4,1,2,4,2,3,1,3,2,3,4,3,4,2,1,4,2,2,3,2,3,1,4,3,4,3,4,1,2,3,2,1,4,1,2,3,3,1,2,1,1,3,2,3,2,3,1,3,4,4,1,3,3,2,4,2,1,4,1,2,3,2,1,3,2,3,1,3,1,2,4,3,4,4,2,2,1,4,2,1,1,3,4,1,1,1,2,1,3,2,4,1,3,2,2,2,2,3,3,2,4,1,4,2,2,3,1,1,1,3,1,1,4,1,3,2,3,1,1,4,2,1,2,1,3,1,3,1,1,1,2,2,2,3,1,2,3,1,3,4,3,3,2,1,4,1,2,4,2,2,1,4,1,2,3,1,4,2,2,3,3,2,1,2,4,3,4,1,1,1,4,4,1,4,3,3,1,4,4,3,1,1,1,4,4,4,3,4,1,4,1,3,2,3,2,4,2,2,2,2,1,1,1,2,3,1,2,2,3,1,3,3,1,3,1,4,2,2,1,2,3,2,1,3,1,4,2,3,2,4,1,1,3,4,3,2,3,1,2,1,4,2,3,1,3,2,2,4,2,3,1,4,4,2,3,3,3,2,1,4,1,1,2,2,3,3,3,4,4,4,1,3,3,2,4,4,4,3,1,3,1,4,2,3,4,2,4,1,3,1,2,4,3,1,3,2,4,3,4,3,2,1,2,3,2,2,3,3,1,3,3,1,4,4,1,3,4,1,4,3,2,4,3,3,1,2,1,3,2,1,3,1,2,3,2,1,3,4,3,2,2,1,4,2,3,3,3,2,4,2,2,3,1,1,4,3,1,2,3,1,3,1,4,3,3,1,1,2,1,3,2,3,2,3,2,1,2,3,4,1,2,2,1,2,3,2,1,1,1,3,3,4,3,2,3,1,3,4,1,4,3,4,2,3,3,3,3,4,2,4,3,4,4,1,3,4,1,3,1,2,2,1,2,1,1,4,1,3,2,3,4,1,2,3,4,4,3,2,2,2,1,4,4,3,3,2,2,1,4,3,2,1,4,4,3,1,4,4,1,4,4,4,3,1,4,2,2,3,4,3,2,2,4,2,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,2,2,2,3,3,3,3,2,1,2,4,1,3,4,4,3,3,4,1,3,2,3,2,4,3,2,1,1,4,2,3,2,2,4,1,3,2,3,2,1,4,3,1,3,3,4,4,2,1,2,3,4,3,3,3,1,3,4,2,1,1,3,2,4,2,3,1,2,4,1,1,3,2,2,4,1,2,1,3,2,4,1,2,2,2,3,2,2,3,1,3,2,3,2,2,3,1,1,2,3,2,3,1,2,3,2,1,3,1,4,3,2,3,4,2,1,4,1,4,3,1,4,2,1,2,4,1,4,2,3,1,2,1,1,2,4,4,1,3,3,4,2,4,2,2,1,3,3,1,4,3,2,4,2,2,2,3,2,4,3,4,1,3,4,2,2,3,4,1,3,2,1,4,1,1,2,4,2,1,3,1,2,1,1,2,1,3,1,3,1,1,2,2,1,1,4,4,2,4,2,1,3,4,3,4,4,2,4,1,4,3,4,4,4,2,4,4,3,1,1,2,4,4,3,4,3,4,3,2,2,3,2,4,2,4,3,1,3,4,3,4,4,2,3,3,1,3,2,1,2,2,3,4,2,1,4,3,2,4,2,3,2,2,3,1,2,3,2,3,4,3,2,3,3,2,4,2,3,4,2,3,2,1,3,4,3,3,2,1,2,4,1,3,2,2,1,3,2,4,3,1,4,2,4,2,4,2,1,1,4,2,1,3,2,3,1,4,2,2,1,2,3,1,3,1,3,1,3,2,1,4,4,1,2,3,1,3,1,2,3,1,2,4,3,1,2,1,3,1,3,3,1,3,3,1,1,3,4,1,3,2,4,4,2,4,3,4,1,3,2,1,4,3,2,2,1,2,2,2,1,1,2,1,1,2,3,2,1,1,2,2,2,2,1,3,1,3,2,3,1,1,2,1,3,1,1,1,1,1,3,4,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,3,1,1,1,2,1,1,1,1,1,1,1,1,1,2,1,1,4,2,1,2,1,1,1,1,3,1,1,2,4,2,1,1,2,1,1,3,2,3,1,1,1,3,1,2,2,4,1,1,1,1,1,2,4,3,1,3,2,3,2,1,2,3,4,3,2,3,3,2,1,2,3,1,1,2,1,3,1,2,2,4,3,1,1,1,1,2,1,2,1,2,1,3,1,4,1,3,1,4,3,2,1,3,1,1,3,2,2,1,4,3,2,2,3,1,3,4,3,2,2,3,2,3,3,2,1,3,4,4,2,2,2,2,2,2,2,3,4,2,1,2,3,2,4,3,3,3,3,4,2,2,2,3,3,3,2,2,3,3,2,1,4,4,3,2,3,3,2,4,2,4,3,2,3,3,3,4,4,2,3,1,1,3,2,2,2,4,3,4,2,3,2,1,2,3,4,2,2,4,4,2,2,3,2,2,2,2,3,4,3,3,1,2,3,2,2,4,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,4,4,3,4,3,4,4,4,3,4,4,4,3,4,4,4,1,1,4,4,1,3,3,4,4,4,1,3,4,4,1,4,3,1,3,4,4,1,1,4,1,4,3,4,3,1,4,4,3,4,4,3,1,2,1,4,1,4,2,3,2,1,3,4,2,3,2,2,3,2,4,4,1,2,2,3,1,2,3,1,4,3,1,4,3,3,2,2,3,4,2,4,1,1,2,2,3,2,4,3,3,4,1,2,3,2,3,1,4,3,3,1,2,4,1,3,2,2,4,3,2,4,4,3,2,1,3,2,4,2,1,4,2,1,4,3,4,3,2,1,4,2,3,3,2,1,3,4,3,2,3,1,4,4,3,2,3,4,3,3,2,1,4,3,1,2,1,3,4,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,2,2,1,1,3,1,2,2,2,1,2,3,4,2,2,1,1,2,2,3,1,1,1,4,1,3,3,2,2,1,2,4,3,4,3,2,4,1,1,3,1,4,2,4,3,1,3,1,1,4,2,3,2,1,3,2,3,4,3,1,4,4,2,2,4,3,1,2,1,1,2,4,1,3,1,4,2,2,1,2,2,3,3,2,4,2,2,1]
    wrongid, bestCount = 0, 0
#======  read data in for loop  ======
    bestModel = ""
    for p in pm:
        count = 0
        ansList = []
        if p[len(p)-6:len(p)] == ".model":
            model = models.Word2Vec.load(pathModel + p)
            for i in range(totalData):
                jsonData = readData(pathData + str(i) + '.json')
                print("Processing file: %d." %i)
        #====== output data =======
                randomNum = True
                if len(jsonData['answer']) != 4:
                    randomNum = False
                    wrongid += 1
                if randomNum == False:
                    ansTag = str(random.randint(1,4))
                elif randomNum == True:
                    ansTag = generateAnswer(jsonData, model)
                # for evaluate correctness
                # if ansTag == str(correctAnsList[i]):
                #     count +=1
                ansList.append(ansTag)
                
                outputList = []
                outputMerge = []
            
                #read question number from csv file
                with open('readNumber.csv', newline= '') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    outputList = list(spamreader)

                    for i in range(0, len(ansList), 1):
                        data = str(outputList[i+1][0])+ str(ansList[i])
                        outputMerge.append(data)

                with open('method1.csv', 'w', newline='') as csvfile:
                    csvfile.write('ID,Answer')
                    csvfile.write('\n')
                    for i in outputMerge:                     
                        csvfile.write(i)    
                        csvfile.write('\n')
            # for evaluate correctness
            # if bestCount < count :
            #     bestCount = count
            #     bestModel = p
            # print("current accuracy: %.3f" %(bestCount/1500))
            # print(bestModel)
            print("=========Finished========")
            print("It took %.2f sec to process" % (time.time() - t))
            del model
if __name__ == "__main__":
    main()

