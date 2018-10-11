
# coding: utf-8

# In[1]:


import json
import os
import unicodedata
import re


t2j_dict = {'C': 'corpus',
'Q': 'question',
'A': 'answer'}


# In[21]:



class txt2json:
    def __init__(self, min_question=0, max_question=1501):
        self.min_q = min_question
        self.max_q = max_question
        self.context = None
        self.ins_line = 0
        self.errCount = 0
        self.json_context = {'corpus':[], 'answer':[], 'question':[]}

    def load_txt(self, file_name, path=os.getcwd(),  encoding='utf-8'):
        print('input : '+os.path.join(path, file_name))
        with open(os.path.join(path, file_name),  encoding=encoding) as f:
            self.context = f.read().splitlines()

    def deal_txt(self, start=None, stop=None):
        self._check_range(start, stop)
        for q_number in range(self.max_q):
            try:
                self.load_txt('CutResult/'+str(q_number)+'.txt')
                if len(self.context)==10:
                    self._assign2json()
                elif len(self.context)!=10:
                    self.errCount+=1
                self.save('JsonResult/'+str(q_number)+'.json')
            except FileNotFoundError:
                print('Check your path, I cant find your file {0} '.format(q_number))
            self.reset()
        print("Error format number: %d.",self.errCount)

    def save(self, file_name, path=os.getcwd()):
#         print('output : '+os.path.join(path, file_name))
        with open(os.path.join(path, file_name), 'w+') as f:
            json.dump(self.json_context, f, ensure_ascii=False)

    def split_txt(self):
        result = []
        for line in self.context:
            splited_line = self._full2half_width(line).split()
            if splited_line:
                result.append(splited_line)
        return result

    def reset(self):
        self.init_json()
        self.context = None
        self.ins_line = 0

    def _assign2json(self):
        while True:
            if self.ins_line >= 9:
                break
            step = self._get_range()
            self._assign_txt(step)
            self.ins_line += step + 1
        if len(self.json_context['answer']) != 4:
            self.init_json()

    def init_json(self):
        self.json_context = self.json_context.fromkeys(t2j_dict.values(), [])

    def _assign_txt(self, step):
        context_line = self.ins_line + 1
        ins = self.get_ins()
#         print('----------ins : {ins}'.format(ins=ins))
        for line in self.context[context_line:context_line+step]:
            line = self._full2half_width(line).split()
#             print('----------line : {line}'.format(line=line))
            line = self._check_UNK(line)
            if ins == 'A':
                if line:
                    self.json_context[t2j_dict[ins]].append(line)
            else:
                self.json_context[t2j_dict[ins]] = line
    ''' remove part of speech '''
    def _ignore_pos(self, txt):
        return txt.split('_')[0]
    ''' check UNK or UNK_pos '''
    def _check_UNK(self, txt_list):
        no_pos_txt_list = [self._ignore_pos(txt) for txt in txt_list]
        while 'UNK' in no_pos_txt_list :
            index = no_pos_txt_list.index('UNK')
            del txt_list[index-1:index+2]
            del no_pos_txt_list[index-1:index+2]
        return txt_list

    def _get_range(self):
        ins = self.get_ins()
        if ins == 'A':
            return 4
        else :
            return 1
    
    ''' return C Q A, striped_half has two format C, Q, A and C_pos, Q_pos, A_pos'''
    def get_ins(self):
        striped_half = self._full2half_width(self.context[self.ins_line]).strip()
        return self._ignore_pos(striped_half)

    def _full2half_width(self, sentence):
        return unicodedata.normalize('NFKC', sentence)
#     number of questions, check the range(start to stop )
    def _check_range(self, start, stop):
        if start:
            self.min_q = start
        if stop:
            self.max_q = stop


# In[22]:


t2j = txt2json()
t2j.deal_txt()


# In[12]:




