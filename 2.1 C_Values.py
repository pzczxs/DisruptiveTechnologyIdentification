# code from: https://github.com/huanyannizu/C-Value-Term-Extraction
import math

noun = ['NN','NNS','NNP','NNPS']  #tags of noun
adj = ['JJ']    #tags of adjective
pre = ['IN']    #tags of preposition


class NoName:
    def word(self, word):
        self.word = word.split('_')[0]
        self.tag = word.split('_')[1]

    def substring(self, sub):
        self.L = len(sub)
        self.words = []
        self.tag = []
        for word in sub:
            self.words.append(word.split('_')[0])
            self.tag.append(word.split('_')[1])
        self.f = 0
        self.c = 0
        self.t = 0

    def CValue_non_nested(self):
        self.CValue = math.log2(self.L) * self.f

    def CValue_nested(self):
        self.CValue = math.log2(self.L) * (self.f - 1 / self.c * self.t)

    def substringInitial(self, f):
        self.c = 1
        self.t = f

    def revise(self, f, t):
        self.c += 1
        self.t += f - t


def term_extracion(tagged_text, ling_filter, max_len, freq_threshold, c_value_threshold):
    """
    extract term from the pos-tagged text, The text format can be see in Turku-tagged.txt.
    Note: ensure the last char tag of each line is not in list noun, adj and pre, or raise the error out of index.
    :param text: list, pos-tagged text
    :param ling_filter: str (Noun or AdjNoun or AdjPrepNoun)
    :param max_len: int, max word number of a term
    :param freq_threshold: int, min term frequency--the term will be drop if its frequency is lower than freq_threshold
    :param c_value_threshold:int, min c-value -- the term will be drop if its c-value is lower than c_value_threshold
    return
        list, each element format is "term\tc_value\tfrequency\ttag"
    """
    # The results are saved in a dictionary named candidate,
    # candidate[m] prints out a list of candidate string objects of length m
    # m ranges from 2 to L
    candidate = dict([(term_len, []) for term_len in range(2, max_len + 1)])
    for sentence in tagged_text:
        sentence = sentence.rstrip('\n').split(' ')
        n_words = len(sentence)
        start = 0
        while start <= n_words - 2:
            i = start
            noun_ind = []
            pre_ind = []
            pre_exist = False
            while True:
                word = NoName()
                word.word(sentence[i])
                if word.tag in noun:
                    noun_ind.append(i)
                    i += 1
                elif (ling_filter == ('AdjNoun' or 'AdjPrepNoun')) and word.tag in adj:
                    word_next = NoName()
                    word_next.word(sentence[i + 1])
                    if word_next.tag in noun:
                        noun_ind.append(i + 1)
                        i += 2
                    elif word_next.tag in adj:
                        i += 2
                    else:
                        end = i + 1
                        break
                elif (ling_filter == 'AdjPrepNoun') and not pre_exist and i != 0 and (word.tag in pre):
                    pre_ind.append(i)
                    pre_exist = True
                    i += 1
                else:
                    end = i
                    break

            if len(noun_ind) != 0:
                for i in list(set(range(start, noun_ind[-1])) - set(pre_ind)):
                    for j in noun_ind:
                        if (j - i >= 1) and (j - i <= max_len - 1):
                            substring = NoName()
                            substring.substring(sentence[i:j + 1])
                            exist = False
                            for ele in candidate[j - i + 1]:
                                if ele.words == substring.words:
                                    ele.f += 1
                                    exist = True
                            if not exist:
                                candidate[j - i + 1].append(substring)
                                substring.f = 1
            start = end + 1

    # Remove candidate strings with low frequency and sort them
    for i in range(2, max_len + 1):
        candidate[i] = [ele for ele in candidate[i] if ele.f > freq_threshold]
        candidate[i].sort(key=lambda x: x.f, reverse=True)

    # Compute C-Value
    term_lst = []
    for l in reversed(range(2, max_len + 1)):
        candi = candidate[l]
        for phrase in candi:
            if phrase.c == 0:
                phrase.CValue_non_nested()
            else:
                phrase.CValue_nested()

            if phrase.CValue >= c_value_threshold:
                term_lst.append(phrase)
                for j in range(2, phrase.L):
                    for i in range(phrase.L - j + 1):
                        substring = phrase.words[i:i + j]
                        for ele in candidate[j]:
                            if substring == ele.words:
                                ele.substringInitial(phrase.f)
                                for m in term_lst:
                                    if ' '.join(ele.words) in ' '.join(m.words):
                                        ele.revise(m.f, m.t)

    term_lst.sort(key=lambda x: x.CValue, reverse=True)

    result = []
    for m in term_lst:
        result.append(f"{' '.join(m.words)}\t{m.CValue}\t{m.f}\t{m.tag}")
    return result


if __name__ == "__main__":
    # 读取标注文件，直接输入函数中
    with open(r"E:\课题组\创新性指标_2\extract_term_from_fulltext\temp_tagged\10166177.txt", 'r', encoding='utf8') as fr:
        text = fr.readlines()
    print(term_extracion(text, 'Noun', 5, 3, 1))
