from C_Values import *
from stanfordcorenlp import StanfordCoreNLP
import pymysql
import zipfile
import re
import time
import json


def tackle_jsondecodererror(sent, tool):
    # 文本过长引起的问题，因此将文本切分成多段即可
    sent_len = len(sent)
    seg_num = sent_len // 30000
    seg_idx = [0]
    for _ in range(1, seg_num + 1):
        seg_idx.append(sent.find(' ', _ * 30000))
    res = []
    for _ in range(1, seg_num + 1):
        sent_seg = sent[seg_idx[_ - 1]: seg_idx[_]]
        res.extend(tool.pos_tag(sent_seg))
    sent_seg = sent[seg_idx[-1]:]
    res.extend(tool.pos_tag(sent_seg))
    return res


def pos(raw_text, tool):
    result = []
    raw_text = re.sub('<[a-zA-Z/\s]+>\n', '', raw_text)
    for para in raw_text.split('\n'):
        try:
            temp = tool.pos_tag(para)
        except json.JSONDecodeError:
            # 句子太长切分成多段
            temp = tackle_jsondecodererror(para, tool)
        temp += [('.', '.')]   # 小bug，每段话最后加个句点，不然那个C_value程序会报错
        result.append(' '.join([f'{word[0].replace(" ", "")}_{word[1]}' for word in temp]))
    return result


nlp_stanford = StanfordCoreNLP('f:/stanford-corenlp-full-2018-10-05', lang='en', timeout=2000000)
conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='sbfgeqiaqg1029', db='patents_view')
cur = conn.cursor()

for year in range(2019, 1977, -1):
    start = time.time()
    print(f'Processing Patents in {year}...')
    # 从数据库选择当年的专利，RE开头或者纯数字
    # candidate_patent就是目标专利数据集既包含目标专利的施引专利信息，也包含被引专利以及被引专利的施引专利信息，记录有23W条
    cur.execute(f'select patent_id from candidate_patent where left(pub_date, 4) = "{year}";')
    patent_in_year = set([ele[0] for ele in cur.fetchall()])

    read_path = f"../extract_fulltext/patent_fulltext/parse_fulltext_{year}.zip"
    zr = zipfile.ZipFile(read_path, 'r')
    count = 0    # 记录匹配专利数目
    for idx, file in enumerate(zr.namelist()[0:]):
        print(f'{idx:2d} \tProcess file {file}...', end='\t')
        text = zr.read(file).decode()
        pat_info = re.findall('<us-patent-grant pat_num="(.*)" pub_date=.*>\n([\s\S]*?)</us-patent-grant>', text)
        del text
        print(f'该文件总专利数目：{len(pat_info)}', end='\t')

        # 从全年的专利全文本数据中，找到我们需要的candidate_patent全文本，并进行术语抽取
        for pat_id, pat_text in pat_info:
            if pat_id in patent_in_year:
                count += 1
                pat_text = pat_text.replace('_', ' ').replace('/', ' ')
                text_tagged = pos(pat_text, nlp_stanford)
            else:
                continue
            with open(f'./temp_tagged/{pat_id}.txt', 'w', encoding='utf8') as fw:
                fw.write('\n'.join(text_tagged))
            term_extracion(text_tagged, 'Noun', 5, 1, 1, f'./temp/{pat_id}.txt')
        print(f'匹配的专利数目：{count}/{len(patent_in_year)}\t time: {(time.time() - start)/60:.2f} min')
    zr.close()


