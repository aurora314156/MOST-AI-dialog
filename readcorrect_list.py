import json

with open('correct_list.json' , 'r') as reader:
    jf = json.loads(reader.read())
    print(jf['correct_answer'][0])
    print(jf['correct_answer'][2])
        