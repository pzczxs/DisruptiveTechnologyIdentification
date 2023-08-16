import pymysql
import pandas as pd
from itertools import combinations


conn = pymysql.Connect(host='localhost', user='root', passwd='sbfgeqiaqg1029', port=3306, db='patents_view')
cur = conn.cursor()

def comb_and_sort(term_set):
    """两两组合，并保证组合内术语按照统一顺序"""
    return set([tuple(sorted(list(ele))) for ele in combinations(term_set, 2)])


# 保存结果
novelty_impact_df = pd.DataFrame(columns=["patent_id", "novelty_single", "novelty_comb", "impact_single", "impact_comb"])

cur.execute("select patent_id, pub_date, term from target_patent where pub_date < 20200101;")
target_patent = pd.DataFrame(cur.fetchall(), columns=["patent_id", "pub_date", "term"])
target_num = len(target_patent)

for idx in range(target_num):
    # 目标专利信息
    patent_id = target_patent.loc[idx, "patent_id"]
    pub_date = str(target_patent.loc[idx, "pub_date"]).replace("-", "")   # 原始数据是datetime格式，转化为字符后为带横杠的日期
    target_term_set = set([int(i) for i in target_patent.loc[idx, "term"].split(",")])
    
    # 计算创新性的候选专利信息
    cur.execute(f"select patent_id, term from candidate_patent where pub_date < {pub_date} and term is not null;")
    candidate_patent = pd.DataFrame(cur.fetchall(), columns=["patent_id", "term"])
    candidate_patent["term"] = candidate_patent["term"].apply(lambda x: set([int(i) for i in x.split(",")] if x else set()))
    
    # 创新性指标_2 = 新术语数 / 总术语数
    candidate_patent["same_term"] = candidate_patent["term"].apply(lambda x: x & target_term_set)
    new_term = target_term_set.copy()
    for ele in candidate_patent["same_term"].tolist():
        if ele:
            new_term -= ele    # 在集合target_term_set中但不在集合ele中的元素的集合
    novelty_single = len(new_term) / len(target_term_set)
    
    # 创新性指标_3 = 新术语组合数 / 总组合数
    candidate_patent["count_same_term"] = candidate_patent["same_term"].apply(lambda x: len(x))
    candidate_patent_filter = candidate_patent[candidate_patent["count_same_term"] >= 2]    # 只保留存在两个共同术语的候选专利
    if len(target_term_set) < 2:
        novelty_comb = 0
    else:
        target_term_set_comb = comb_and_sort(target_term_set)
        candidate_patent_filter["term_comb"] = candidate_patent_filter["term"].apply(lambda x: comb_and_sort(x) if len(x) >=2 else set())
        candidate_patent_filter["same_term_comb"] = candidate_patent_filter["term_comb"].apply(lambda x: target_term_set_comb & x)
        new_term_comb = target_term_set_comb.copy()
        for ele in candidate_patent_filter["same_term_comb"].tolist():
            if ele:
                new_term_comb -= ele
        novelty_comb = len(new_term_comb) / len(target_term_set_comb)
    
    
    # 计算影响力的候选专利信息
    cur.execute(f"select patent_id, term from candidate_patent where pub_date > {pub_date} and term is not null;")
    candidate_patent = pd.DataFrame(cur.fetchall(), columns=["patent_id", "term"])
    if len(candidate_patent) == 0:
        # 没有前向专利，所以直接为0
        novelty_impact_df = novelty_impact_df.append(
            {"patent_id": patent_id, "novelty_single": novelty_single, "novelty_comb": novelty_comb, 
             "impact_single": 0, "impact_comb": 0}, ignore_index=True)
        # print()
        continue
    
    
    candidate_patent["term"] = candidate_patent["term"].apply(lambda x: set([int(i) for i in x.split(",")] if x else set()))
    
    # 影响力指标_2 = （术语复用的专利数目 / 总候选专利的数目） * （被复用的术语数目 / 目标专利总术语数）
    # 可以理解为专利术语被后续专利复用的概率，同时考虑了术语数量的影响，公式后半部分可以看做是对术语数量影响的修正。
    candidate_patent["same_term"] = candidate_patent["term"].apply(lambda x: x & target_term_set)
    candidate_patent["count_same_term"] = candidate_patent["same_term"].apply(lambda x: len(x))
    fwd_part = sum(candidate_patent["count_same_term"] >= 1) / len(candidate_patent)
    reuse_term = set()
    for ele in candidate_patent["same_term"].tolist():
        if ele:
            reuse_term |= ele
    bwd_part = len(reuse_term) / len(target_term_set)
    impact_single = fwd_part * bwd_part
    
    # 影响力指标_3 = （术语组合复用的专利数目 / 总候选专利的数目） * （被复用的术语组合数目 / 目标专利总术语组合数）
    candidate_patent_filter = candidate_patent[candidate_patent["count_same_term"] >= 2]
    fwd_part_comb = sum(candidate_patent["count_same_term"] >= 2) / len(candidate_patent)
    if len(target_term_set) < 2:
        impact_comb = 0
    else:
        
        target_term_set_comb = comb_and_sort(target_term_set)
        candidate_patent_filter["term_comb"] = candidate_patent_filter["term"].apply(lambda x: comb_and_sort(x) if len(x) >=2 else set())
        candidate_patent_filter["same_term_comb"] = candidate_patent_filter["term_comb"].apply(lambda x: target_term_set_comb & x)
        reuse_term_comb = set()
        for ele in candidate_patent_filter["same_term_comb"].tolist():
            if ele:
                reuse_term_comb |= ele
        bwd_part_comb = len(reuse_term_comb) / len(target_term_set_comb)
        impact_comb = fwd_part_comb * bwd_part_comb
        
    print(f"进度：{idx+1}/{target_num}, 目标专利：{patent_id}, "
          f"novelty_single: {novelty_single: .4f}, novelty_comb: {novelty_comb: .4f}, "
          f"impact_single: {impact_single: .4f}, impact_comb: {impact_comb: .4f}")
    
    novelty_impact_df = novelty_impact_df.append({"patent_id": patent_id, 
                                                  "novelty_single": novelty_single, "novelty_comb": novelty_comb, 
                                                  "impact_single": impact_single, "impact_comb": impact_comb
                                                 }, ignore_index=True)
    
    
novelty_impact_df.to_excel("novelty_impact.xlsx")