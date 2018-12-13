
import math
import numpy as np
from gensim.models import word2vec
from gensim import models
from scipy import spatial
from random import randint
import operator
import time

class race():
    def __init__(self, C, Q, A):
        self.C = C.strip()
        self.Q = Q.strip()
        self.A = A
        self.model = models.Word2Vec.load("5cbowword2vec.model")
    
    def main(self):
        ans = ['A','B','C','D']
        finalAns, data = [], {}

        data['corpus'] = self.removeStopWords(self.C)
        data['question'] = self.removeStopWords(self.Q)
        data['answer'] = self.removeStopWords(self.A)
        # avoid recognition error
        if len(data['corpus']) == 0 or len(data['question']) == 0 or len(data['answer']) <4:
            print("This file has some recognition error.")
            for i in range(4):
                finalAns.append(ans[randint(0, 3)])
            return finalAns[0], finalAns[1], finalAns[2], finalAns[3]
        else:
            print("Starting process answer.")
            method1 = self.method1(data)
            method2 = self.method2(data)
            method3 = self.method3(data)
            voting = self.voting(method1, method2, method3)
            return method1, method2, method3, voting
            
    def removeStopWords(self,words):
        if isinstance(words,list):
            temp =[]
            for k in words:
                for c in k.strip():
                    if c >= '\uff10' and c <= '\uff19' or c >= '\uff21' and c<= '\uff3A' or c>= '\uff41' and c<= '\uff5A':
                        pass
                    elif c > '\u9fff' or c < '\u4e00' and c > '\u2fdf' or c < '\u2e80' and c > '\u4dbf' or c < '\u3400' and c < '\u0030' and c > '\u0039':
                        k = k.replace(c," ")
                temp.append(k.split())
            return temp
        else:
            for c in words:
                if c >= '\uff10' and c <= '\uff19' or c >= '\uff21' and c<= '\uff3A' or c>= '\uff41' and c<= '\uff5A':
                    pass
                elif c > '\u9fff' or c < '\u4e00' and c > '\u2fdf' or c < '\u2e80' and c > '\u4dbf' or c < '\u3400' and c < '\u0030' and c > '\u0039':
                    words = words.replace(c," ")
            return words.split()

    def method1(self,data):
        CQ_con = np.zeros(250, dtype = float)
        A_con = np.zeros((250, 250), dtype = float)
        anslist = ['A', 'B', 'C', 'D']
        CQ_list = data['corpus']
        CQ_list.extend(data['question'])

        for word in CQ_list:
            try:
                vector = self.model[word]
            except KeyError as e:
                continue
            for i in range(250):
                CQ_con[i] += vector[i]
                
        for i in range(250):
            CQ_con[i] /= 250

        for j in range(0, 4):
            for word in data['answer'][j]:
                try:
                    vector = self.model[word]
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

    def method2(self,data):
        C_con = np.zeros(250, dtype = float)
        QA_con = np.zeros((250, 250), dtype = float)
        anslist = ['A', 'B', 'C', 'D']
        C_list = data['corpus']
        QA_list = []               
        
        for j in range (0, 4):           
            QA_list.append(data['question'])      
            
        for word in C_list:
            try:
                vector = self.model[word]
            except KeyError as e:
                continue
            for i in range(250):
                C_con[i] += vector[i]
        
        for i in range(250):
            C_con[i] /= 250

        for j in range(0, len(data['answer'])): 
            QA_list[j].extend(data['answer'][j])
            for word in QA_list[j]:
                try:
                    vector = self.model[word]
                except KeyError as e:
                    continue
                for i in range(250):
                    QA_con[j][i] += vector[i]
                for i in range(250):
                    QA_con[j][i] /= 250

        ini = 0
        high_cq = 0
        i = 0
        ans = 0
        
        for qa in QA_con:
            cos = 1 - spatial.distance.cosine(C_con, qa)
            if cos > ini:
                ini = cos
                high = qa
                ans = i
            i += 1
            
        return anslist[ans]

    def method3(self,data):

        questionList = data['question']
        answerList = data['answer']

        idfTable = {}
        for q in questionList:
            if q not in idfTable:
                idfTable[q] = 1
            else:
                idfTable[q] += 1

        finalAns, flag = 0, 0
        highestScore = 0
        ans = ['A','B','C','D']

        for A in answerList:
            currentAnsScore = 0
            for q in questionList:
                align = self.align(self.model, 0.1, q, A)
                currentAnsScore += math.log(((30 - idfTable[q]+ 0.5) / ( idfTable[q]+0.5 ))) * align
            if highestScore <= currentAnsScore:
                highestScore = currentAnsScore
                finalAns = flag
            flag += 1
    
        return ans[finalAns]
    
    # calculate align
    def align(self, model, x, q, A):
        
        termScoreTable = self.similarTermScoreTable(model, q, A)
        # if similar terms bigger than one, add neg to align
        if len(termScoreTable) > 1:
            align = termScoreTable[0] + x * termScoreTable[1]
        else:
            align = termScoreTable[0]
        
        return align
    
    # calculate consine similarity table
    def similarTermScoreTable(self, model, q, A):

        similarTermScore = []
        for cutWord in A:
            # if word not in model append 0
            try:
                model.similarity(q, cutWord)
            except KeyError:
                similarTermScore.append(0)
                continue
            else:
                similarTermScore.append(model.similarity(q, cutWord))

        similarTermScore.sort(reverse=True)
        
        return similarTermScore

    def voting(self,method1, method2, method3):
        ans = ['A','B','C','D']
        v = {}
        v[method1], v[method2], v[method3] = 0, 0, 0

        for a in range(4):
            if method1 == ans[a]:
                v[method1] += 1
        for a in range(4):
            if method2 == ans[a]:
                v[method2] += 1
        for a in range(4):
            if method3 == ans[a]:
                v[method3] += 1
        return max(v.items(), key=operator.itemgetter(1))[0]
    

a,b,c,d = race('學年 度 大學 學測 共有 十二萬 八千 七百五十 九人 報名 身心。障礙 考生 有 三 百五十 九人 其。中 一 百二十 九人 在 台大 身障 考生。應考 台大。註冊組 主任 洪泰雄 表示 今年 身障 考生 中 注意。患者 最多 有 三十 八人 其次 是 學習 障礙 有二 十 二人 失蹤 生死 九人 輪椅族。刑期 外 考生 申請 使用 口袋 和 事蹟 放大 滑鼠 等 設備 都。有。一 個 就是 破。自己 最。主要 是 他 新 曉波 然後。他 保證 鏡頭 跟 螢幕 成為。一 個 機器 同學。在 媽媽 的 陪同 下 到 台大 查看 考場 因為 如果 是 申請 考卷 字體 放大 延長 考試 時間 等 服務 我。媽媽 說 孩子 視力 有些 學校 會 比較 辛苦 因此 只要 盡力 就 好 那。他 的 成績 一般。說 來 比喻 是 比不上 因為。它 可能 付出 那麼多 也 不見得 能夠 得到 更 好 的 成果 對。那 所以 就是 盡力 就 好 強烈。大陸 冷氣團 南下 考試 當天 氣溫 偏低 大考。中心 提醒 考生 可 攜帶 口罩 手套 圍巾 或是。使用 電子。是 傳統型 應。考慮 配合。檢查 人員 檢查 經過。封信。在 臺北 ','請問 關於 報導 中 和 同學 的 市樹 her 證券 ',['因 輕微 若是 申請 考試 試卷 字體 放大 至 ','三 <UNK> 和 同學 的 考試 時間 與 一般生 一樣 ', '和 她 的 媽媽 希望 藉由 身障 考生 資格 讓 兒子 能夠 加上 台大 ']).main()
print("----1",a,b,c,d)
a,b,c,d = race('學年 度 大學 學測 共有 十二萬 八千 七百五十 九人 報名 身心。障礙 考生 有 三 百五十 九人 其。中 一 百二十 九人 在 台大 身障 考生。應考 台大。註冊組 主任 洪泰雄 表示 今年 身障 考生 中 注意。患者 最多 有 三十 八人 其次 是 學習 障礙 有二 十 二人 失蹤 生死 九人 輪椅族。刑期 外 考生 申請 使用 口袋 和 事蹟 放大 滑鼠 等 設備 都。有。一 個 就是 破。自己 最。主要 是 他 新 曉波 然後。他 保證 鏡頭 跟 螢幕 成為。一 個 機器 同學。在 媽媽 的 陪同 下 到 台大 查看 考場 因為 如果 是 申請 考卷 字體 放大 延長 考試 時間 等 服務 我。媽媽 說 孩子 視力 有些 學校 會 比較 辛苦 因此 只要 盡力 就 好 那。他 的 成績 一般。說 來 比喻 是 比不上 因為。它 可能 付出 那麼多 也 不見得 能夠 得到 更 好 的 成果 對。那 所以 就是 盡力 就 好 強烈。大陸 冷氣團 南下 考試 當天 氣溫 偏低 大考。中心 提醒 考生 可 攜帶 口罩 手套 圍巾 或是。使用 電子。是 傳統型 應。考慮 配合。檢查 人員 檢查 經過。封信。在 臺北 ','請問 關於 報導 中 和 同學 的 市樹 her 證券 ',['e 獨自 前往 台大 查看 考場 ','因 輕微 若是 申請 考試 試卷 字體 放大 至 ','三 <UNK> 和 同學 的 考試 時間 與 一般生 一樣 ', '和 她 的 媽媽 希望 藉由 身障 考生 資格 讓 兒子 能夠 加上 台大 ']).main()
print("----2",a,b,c,d)
a,b,c,d = race('學年 傳統型 應。考慮 配合。檢查 人員 檢查 經過。封信。在 臺北','',['e 獨自 前往 台大 查看 考場 ','因 輕微 若是 申請 考試 試卷 字體 放大 至 ','三 <UNK> 和 同學 的 考試 時間 與 一般生 一樣 ', '和 她 的 媽媽 希望 藉由 身障 考生 資格 讓 兒子 能夠 加上 台大 ']).main()
print("----3",a,b,c,d)
a,b,c,d = race('','學年 度 大學 學測 共有 十二萬 八千 七百五十 九人 報名 身心。障礙 考生 有 三 百五十 九人 其。中 一 百二十 九人 在 台大 身障 考生。應考 台大。註冊組 主任 洪泰雄 表示 今年 身障 考生 中 注意。患者 最多 有 三十 八人 其次 是 學習 障礙 有二 十 二人 失蹤 生死 九人 輪椅族。刑期 外 考生 申請 使用 口袋 和 事蹟 放大 滑鼠 等 設備 都。有。一 個 就是 破。自己 最。主要 是 他 新 曉波 然後。他 保證 鏡頭 跟 螢幕 成為。一 個 機器 同學。在 媽媽 的 陪同 下 到 台大 查看 考場 因為 如果 是 申請 考卷 字體 放大 延長 考試 時間 等 服務 我。媽媽 說 孩子 視力 有些 學校 會 比較 辛苦 因此 只要 盡力 就 好 那。他 的 成績 一般。說 來 比喻 是 比不上 因為。它 可能 付出 那麼多 也 不見得 能夠 得到 更 好 的 成果 對。那 所以 就是 盡力 就 好 強烈。大陸 冷氣團 南下 考試 當天 氣溫 偏低 大考。中心 提醒 考生 可 攜帶 口罩 手套 圍巾 或是。使用 電子。是 傳統型 應。考慮 配合。檢查 人員 檢查 經過。封信。在 臺北 ',['e 獨自 前往 台大 查看 考場 ','因 輕微 若是 申請 考試 試卷 字體 放大 至 ','三 <UNK> 和 同學 的 考試 時間 與 一般生 一樣 ', '和 她 的 媽媽 希望 藉由 身障 考生 資格 讓 兒子 能夠 加上 台大 ']).main()
print("----3",a,b,c,d)

