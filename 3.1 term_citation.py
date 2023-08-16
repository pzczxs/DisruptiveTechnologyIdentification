"""根据专利引用关系构建术语引用"""

import pandas as pd
import pymysql
import itertools
from dateutil.relativedelta import relativedelta
import re
from itertools import product
import time

conn = pymysql.Connect(host='localhost', user='root', passwd='sbfgeqiaqg1029', port=3306, db='patents_view')
cur = conn.cursor()

select = """
SELECT patent_id, citation_id, date_1, date_2, term_1, term_2 
from us_patent_citation_new;
"""
cur.execute(select)
data = cur.fetchall()
data_len = len(data)

create_table = """
CREATE TABLE IF NOT EXISTS term_citation (
	`term_1` VARCHAR(25),
	`term_2` VARCHAR(25),
	`date_1` DATE,
	`date_2` DATE
);
"""

cur.execute('create_table')
insert = 'insert into term_citation (term_1, term_2, date_1, date_2) values (%s, %s, %s, %s)'
start = time.time()
for idx, ele in enumerate(data):
    patent_id, citation_id, date_1, date_2, term_1, term_2 = ele

    if not term_1:
        continue
    if not term_2:
        continue
    term_1 = term_1.strip().split(',')
    term_2 = term_2.strip().split(',')
    term_citation = product(term_1, term_2)
    for t1, t2 in term_citation:
        cur.execute(insert, (t1, t2, date_1, date_2))
    conn.commit()
    
    if idx % 1000 == 0:
        end = time.time()
        print(f'进度：{idx}/{data_len}', end='\t')
        print(f'术语引用关系数目：{len(term_1)*len(term_2)}\t所需时间: {end - start: .2f} s')
        start = time.time()