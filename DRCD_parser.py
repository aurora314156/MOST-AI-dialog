
import json
from pprint import pprint

with open('./DRCD_test.json') as f:
    data = json.load(f)

#outputFile = open('./dataset.txt', 'w')
count = 1

dataLength = len(data["data"])
for i in range(dataLength):
    paragraphLength = len(data["data"][i]["paragraphs"])
    for j in range(paragraphLength):
        context = data["data"][i]["paragraphs"][j]["context"]
        questionLength = len(data["data"][i]["paragraphs"][j]["qas"])
        for k in range(questionLength):
            outputFile = open(("testDataset/dataset" + str(count) + ".txt"), "w")
            outputFile.write("C\n")
            outputFile.write(context)
            outputFile.write ("\nQ\n")
            outputFile.write (data["data"][i]["paragraphs"][j]["qas"][k]["question"])
            outputFile.write ("\nA\n")
            outputFile.write (data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0]["text"])
            outputFile.write ("\n")
            count = count + 1
