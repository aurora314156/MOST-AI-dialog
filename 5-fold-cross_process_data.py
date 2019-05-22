

doclen = 1500
for i in range(1,(doclen+1)):
    with open('6/' + str(i) +'.json', 'r') as file:
        fileData = file.read()
        with open('AllMostData/'+ str(7050 + i) + '.json', 'w') as j:
            j.write(fileData)
