#!coding:utf-8
import re

import jieba

#正则表达式处理法
def textParse(str_doc):
    str_doc = re.sub('\u3000', '', str_doc)
    return str_doc

#去除停用词和数字、空字符、长度为1的字符
def rm_tokens(words, stwlist):
    words_list = list(words)
    stop_words = stwlist
    for i in range (words_list.__len__())[::-1]:
        if words_list[i] in stop_words:
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
        elif len(words_list[i]) == 1 :
            words_list.pop(i)
        elif words_list[i] == " ":
            words_list.pop(i)
    return words_list

#加载停用词词典
def get_stop_words(path = r'C:\Users\15561\PycharmProjects\pres\FileLoader\StopWords.txt'):
    file = open (path, 'r', encoding='utf-8').read().split('\n')
    return set(file)

#清洗单条文本数据
def seg_doc (str_doc):
    sent_list = str_doc.split('\n')
    sent_list = map(textParse, sent_list)
    stwlist = get_stop_words()
    word_2dlist = [rm_tokens(jieba.cut(part),stwlist) for part in sent_list]
    word_list = sum(word_2dlist, [])
    return word_list