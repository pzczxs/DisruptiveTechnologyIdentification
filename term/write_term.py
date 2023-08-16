"""
过滤术语，同时写入 candidate_patent, target_patent, us_patent_citation_new 表中
"""

import pymysql
import re
import os
from nltk import stem
import pandas as pd


conn = pymysql.Connect(host='localhost', user='root', passwd='sbfgeqiaqg1029', port=3306, db='patents_view')
cur = conn.cursor()


def filter_term(term):
    if re.search('[0-9]', term):
        return 0

    # 剔除所有单词都不超过3个字符的术语
    word_lst = term.split()
    word_len_check = [True for word in word_lst if len(word) > 3]
    if True not in word_len_check:
        return 0
    # 剔除存在单词字符长度超过45的术语
    word_len_check = [True for word in word_lst if len(word) > 45]
    if True in word_len_check:
        return 0

    # 剔除全部单词均为一样的术语
    word_set = set(word_lst)
    if (len(word_set) == 1) and (len(word_lst) != len(word_set)):
        return 0

    # 单词中只能包含英文字母及最多一个横杆
    word_char_check = [False for word in word_lst if not (re.search('^[A-Za-z\-]+$', word) and word.count('-') <= 1)]
    if False in word_char_check:
        return 0

    return 1

# 根据 petent_id 更新 term, term_len
# term 来源于文件夹 ./extracted_term
# 生成对应的 term_dict 文件
print("开始运行...")
term_dict = dict()
path = r'./term/extracted_term'
file_lst = os.listdir(path)
update_sql = 'UPDATE candidate_patent set term = %s, term_num = %s where patent_id = %s;'
for idx, file in enumerate(file_lst):
    term_idx_lst = []
    with open(path + '/' + file, 'r', encoding='utf8') as fr:
        text = fr.readlines()

    count = 0
    for line in text:
        term = line.split('\t')[0].strip().lower()
        c_val = float(line.split('\t')[1].strip())
        if c_val < 10 and count >= 5:
        # if c_val <= 10 and count >= 5:
            continue
        # 是否保留该术语
        retain = filter_term(term)
        if retain:
            count += retain
            # print(term)
            
            # 词原化
            term = ' '.join([stem.WordNetLemmatizer().lemmatize(word) for word in term.split()])
            
            if term not in term_dict:
                term_dict[term] = len(term_dict)
            term_idx_lst.append(str(term_dict[term]))

    cur.execute(update_sql, [','.join(term_idx_lst), len(term_idx_lst), file[:-4]])

    # print(f'专利文档：{file}\t原术语数量：{len(text)}\t筛选后术语数量：{count}')
    if idx % 10000 == 0:
        print(idx, len(term_dict))
        conn.commit()

fw = open(r'./term/term_dict.txt', 'w', encoding='utf8')
for k, v in term_dict.items():
    fw.write(f'{v}\t{k}\n')
fw.close()


# 同时更新 target_patent，us_patent_citation_new 
update_sql = """
UPDATE target_patent, candidate_patent
set target_patent.term = candidate_patent.term
where target_patent.patent_id = candidate_patent.patent_id;
"""
cur.execute(update_sql)

# UPDATE  us_patent_citation_new set term_1=null, term_2=null;
update_sql = """
UPDATE us_patent_citation_new, candidate_patent
set term_1 = term
where us_patent_citation_new.patent_id = candidate_patent.patent_id;
"""
cur.execute(update_sql)

update_sql = """
UPDATE us_patent_citation_new, candidate_patent
set term_2 = term
where us_patent_citation_new.citation_id = candidate_patent.patent_id;
"""
cur.execute(update_sql)

conn.commit()

# 术语统计
cur.execute("select patent_id, pub_date, term from candidate_patent where pub_date <= 20191231 and term is not null;")
term = pd.DataFrame(cur.fetchall(), columns=["patent_id", "pub_date", "term"])
term['term_num'] = term.term.apply(lambda x: x.count(",")+1)
term.describe()

