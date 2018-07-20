# MOST-AI-dialog
This repository is for MOST AI dialog contest.

## format output using

1. 需要把圖中的資料都按照對應的資料夾路徑放置

2. CQA資料夾底下新增3個folder,  cut_all_false, cut_all_true, cut_for_search  分別為三種不同jieba cut產出的結果存放位置

3. 執行python3 formatOutput.py 產生資料集結果 (每次產生新結果時，會先把舊結果全部刪除)

4. (重要) 若有新增資料集，直接新增到CQA.txt，建議照著格式新增到檔案開始的地方，因為最後一筆資料不會輸出到Result裡

5. load 不同的word2vec model 需找到 models.Word2Vec.load 更改後方路徑參數

6. 最終同一層的檔案名稱有: CQA, gigaword, jieba_setting, wiki, CQA.txt, formatOutput.py

### Update history

`2018/06/03 upload first version`

`2018/06/04 modify algorithm with sum and avg`

`2018/06/06 modify algorithm into C, Q+A two part`

`2018/06/16 add algorithm of C+Q, A two part`

`2018/06/25 generate random answer`

`2018/07/20 change output format to csv file, modify gitignore`



