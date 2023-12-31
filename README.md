# DisruptiveTechnologyIdentification
基于专利术语的颠覆性技术识别

## 1. Introduction
### 1.1 数据源
（1）USPTO下载的1976-2019年的专利全文本数据，比如2004年数据的下载链接为（其他年份只需修改一下年份即可）：
https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/2004/

（2）从 PatentsView 数据库中筛选CPC分类号为Y02E 10/545（微晶硅光伏电池）、Y02E 10/546（多晶硅光伏电池）、Y02E 10/547（单晶硅光伏电池）和Y02E 10/548（非晶硅光伏电池）的专利，专利公开年份限定为1976-2019年，所搜集的目标专利类型均为已授权的发明专利。经过筛选后，得到4403个目标专利。

候选专利集既包含目标专利的施引专利信息，也包含被引专利以及被引专利的施引专利信息。

### 1.2 术语抽取
C-Value抽取术语过程详见代码```C_Values.py```和```extract_term_corenlp.py```文件，主要分为以下几步：读取全文本、切分段落、文本词性标注、抽取候选术语集、计算C-value值、术语筛选等。术语抽取后，得到230696个专利原始术语文件（一个专利对应一个文件），详见```term/extracted_term/```文件夹。

剔除一些噪声术语，详见```term/write_term.py```程序。主要操作包括：进行词形还原、剔除包含数字等其他非英语字符的术语、剔除所有单词不超过三个字符的术语、剔除存在单词字符长度超过45的术语、剔除全部单词均为一样的术语、剔除单词存在两个以上横杠的术语等。处理后，得到术语总计451100个，详见```term/term_dict.txt```文件，并同时将专利对应的术语写入数据库中。

### 1.3 指标计算
（1）根据专利引用关系，构建术语引用关系，详见```term_citation.py```程序。
如：PatA术语有T1，T2；PatB有术语T3，T4；PatB引用了PatA，那么术语引用就有T3引用T1和T2、T4引用T1和T2这四种引用关系（即采用笛卡尔积形式）。

（2）计算专利公开5年后的影响力指标，详见```disruption_indicator.py```文件。

（3）计算计算基于术语创新和术语复用的四个指标，详见```combination_indicator.py```文件。

（4）将上述数据按照patent_id整合起来得到最终的特征数据```model_data.xlsx```。根据Netmet et al. (2012)和Sun et al. (2021)，将1977-1996年期间公开的824个专利作为参加训练模型的数据，这些专利中有104个专利被上述两个研究确定为具有重大颠覆的专利，剩余的720个专利被确定为非颠覆性专利。

### 1.4 模型训练及预测
详见```model.py```文件。主要步骤：读取model_data数据，1977-1996年期间824个专利进行模型训练，绘制DCA曲线以确定最优分类阈值。确定分类阈值后，对1997-2014年专利进行预测，最后的预测结果```patent_disruption_result.xlsx```。
## 2. References
