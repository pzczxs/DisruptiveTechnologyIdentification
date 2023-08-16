import networkx as nx
import pymysql
import matplotlib.pyplot as plt
import pandas as pd
import time

# 数据恢复到patents_view数据库中的命令：mysql -u root -p patents_view term_citation < f:/term_citation.sql
# 修改数据库参数
conn = pymysql.Connect(host='localhost', user='root', passwd='sbfgeqiaqg1029', port=3306, db='patents_view')
cur = conn.cursor()

for year in range(2010, 2015):
    # 当年的术语节点
    select = f"select distinct term_1 from term_citation where date_1 between {str(year)}0101 and {str(year)}1231;"
    print(f'{year} 年术语数目：{cur.execute(select)}')

    data = cur.fetchall()
    term_result = pd.DataFrame(index=[ele[0] for ele in data], columns=range(year+1, year+6))

    # 构建i年内的术语引用网络
    print(f"正在计算 {year} 年术语颠覆性指标...")
    DG = nx.DiGraph()
    start_date = str(year) + '0101'
    for i in range(1, 6):
        end_date = str(year+i) + '1231'
        select = f"select distinct term_2, term_1 from term_citation where date_1 between {start_date} and {end_date};"
        print(f'\t{year} 年 - {year+i} 年术语引用关系数目：{cur.execute(select)}')
        DG.add_edges_from(cur.fetchall())    # 箭头从第一个元素指向第二个元素，前者引用后者
        print('\t术语网络构建完毕！')

        start = time.time()
        for ele in data:
            term = ele[0]
            bw_term = set(DG.predecessors(term))  # 返回前向引用
            fw_term = set(DG.successors(term))  # 返回后向引用

            ni, nj = 0, 0
            for ft in fw_term:
                temp = set(DG.predecessors(ft))
                check = temp & bw_term
                if len(check) >= 100:
                    nj += 1
                else:
                    ni += 1

            nk_set = set()
            for bt in bw_term:
                temp = set(DG.successors(bt))
                nk_set |= temp
            nk_set = nk_set.difference(fw_term)
            nk = len(nk_set)

            if ni + nj + nk == 0:
                index = 0
            else:
                index = (ni - nj) / (ni + nj + nk)
            term_result.loc[term, year+i] = index
    term_result.to_excel(f"./indicators/disruption/term_disruption_{year}.xlsx")

    # term_result = pd.read_excel(f"./disruption/term_disruption_{year}.xlsx", index_col=0)
    # term_result.index = term_result.index.astype("str")
    select = f"select patent_id, term, pub_date from target_patent where pub_date between '{year}0101' and '{year}1231';"
    cur.execute(select)
    tar_data = cur.fetchall()

    # 计算专利公开1-5年后的5个影响力指标，最后实际上只选取了5年后的那个指标
    pat_result = pd.DataFrame(index=[ele[0] for ele in tar_data], columns=range(year+1, year+6))
    for col in term_result.columns:
        # characteristics scores and scales 特征分数和尺度法
        x1 = term_result[col].mean()
        class_1 = term_result[term_result[col] < x1]
        class_1_term = set(class_1.index)
        print(f'\tclass_1术语数量：{len(class_1):5d}\t\t对应指标区间 [-1, {x1:.4f})')

        rest = term_result[term_result[col] >= x1]
        x2 = rest[col].mean()
        class_2 = rest[rest[col] < x2]
        class_2_term = set(class_2.index)
        print(f'\tclass_2术语数量：{len(class_2):5d}\t\t对应指标区间 [{x1:.4f}, {x2:.4f})')

        rest = rest[rest[col] >= x2]
        x3 = rest[col].mean()
        class_3 = rest[rest[col] < x3]
        class_3_term = set(class_3.index)
        print(f'\tclass_3术语数量：{len(class_3):5d}\t\t对应指标区间 [{x2:.4f}, {x3:.4f})')

        class_4 = rest[rest[col] >= x3]
        class_4_term = set(class_4.index)
        print(f'\tclass_4术语数量：{len(class_4):5d}\t\t对应指标区间 [{x3:.4f}, 1]')

        for patent_id, term, pub_date in tar_data:
            term_set = set([i for i in term.strip().split(',')])
            temp = class_4_term & term_set
            if temp:
                pat_disruption = class_4.loc[list(temp), col].mean()
                pat_result.loc[patent_id, col] = pat_disruption
                continue

            temp = class_3_term & term_set
            if temp:
                pat_disruption = class_3.loc[list(temp), col].mean()
                pat_result.loc[patent_id, col] = pat_disruption
                continue

            temp = class_2_term & term_set
            if temp:
                pat_disruption = class_2.loc[list(temp), col].mean()
                pat_result.loc[patent_id, col] = pat_disruption
                continue

            temp = class_1_term & term_set
            if temp:
                pat_disruption = class_1.loc[list(temp), col].mean()
                pat_result.loc[patent_id, col] = pat_disruption
                continue
            pat_result.loc[patent_id, col] = 0  # 兜底
    pat_result.to_excel(f"./indicators/disruption/pat_disruption_{year}.xlsx")
