#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
import time
from flashtext.keyword import KeywordProcessor  # 源码被修改过，文件为keyword.py
from datetime import datetime
import warnings
import configparser
from pprint import pprint
from Str_proprocess import Str_replace
warnings.filterwarnings("ignore")

#设定设备类型范围
modalities=['CT',"MR","DX","MG"]
# 读取知识图谱和预处理模板
def GetLevelDic(firstdf):  
    """从excel部位字典中读取知识图谱"""
    partdic = {}
    i = 0
    axis_start = [0]*6
    axis_end = [0]*6
    hiatus = firstdf[(firstdf['起始坐标'] == np.nan) | (firstdf['终止坐标'] == np.nan)]
    if len(hiatus) > 0:
        print("axis loss:", hiatus)
    for index, row in firstdf.iterrows():
        Partlist = []
        parts = [[], [], [], [], [], []]
        if row["六级部位"] is not np.nan:
            parts[5] = row['六级部位'].split("|")
            Partlist.append(parts[5][0])
            axis_start[5] = row['起始坐标']
            axis_end[5] = row['终止坐标']
        if row["五级部位"] is not np.nan:
            parts[4] = row['五级部位'].split("|")
            Partlist.append(parts[4][0])
            temp = firstdf[firstdf['五级部位'] == row['五级部位']]
            axis_start[4] = temp['起始坐标'].min()
            axis_end[4] = temp['终止坐标'].max()
        if row["四级部位"] is not np.nan:
            parts[3] = row['四级部位'].split("|")
            Partlist.append(parts[3][0])
            temp = firstdf[firstdf['四级部位'] == row['四级部位']]
            axis_start[3] = temp['起始坐标'].min()
            axis_end[3] = temp['终止坐标'].max()
        if row["三级部位"] is not np.nan:
            parts[2] = row['三级部位'].split("|")
            Partlist.append(parts[2][0])
            temp = firstdf[firstdf['三级部位'] == row['三级部位']]
            axis_start[2] = temp['起始坐标'].min()
            axis_end[2] = temp['终止坐标'].max()
        if row["二级部位"] is not np.nan:
            parts[1] = row['二级部位'].split("|")
            Partlist.append(parts[1][0])
            temp = firstdf[firstdf['二级部位'] == row['二级部位']]
            axis_start[1] = temp['起始坐标'].min()
            axis_end[1] = temp['终止坐标'].max()
        if row["一级部位"] is not np.nan:
            parts[0] = row['一级部位'].split("|")
            Partlist.append(parts[0][0])
            axis_start[0] = firstdf['起始坐标'].min()
            axis_end[0] = firstdf['终止坐标'] .max()
        for i in range(6):
            newlist = Partlist[::-1]
            newlist = newlist[:i+1]
            if parts[i] != []:
                # print(newlist,axis_start,axis_end)
                newlist.append((axis_start[i], axis_end[i]))
                partdic[tuple(newlist)] = parts[i]
    # print(partdic)
    return partdic

def read_knowledgegraph(modality:str):
    """读取部位知识图谱"""
    Parts_Knowledgegraph = pd.read_excel("knowledge_graph\\Knowledgegraph.xlsx", sheet_name=modality)
    Parts_Knowledgegraph=Parts_Knowledgegraph.fillna("")
    modality_Knowledgegraph = []
    firstlevel = set(Parts_Knowledgegraph["分类"].tolist())
    for firtpart in firstlevel:
        temp = Parts_Knowledgegraph[Parts_Knowledgegraph["分类"] == firtpart]
        modality_Knowledgegraph.append(GetLevelDic(temp))
    return modality_Knowledgegraph
#读取特殊检查字典
def Get_special_exam(modality:str):
    """从excel部位字典中读取keyword."""
    exam=pd.read_excel("knowledge_graph\\exam_special.xlsx", sheet_name=modality)
    exam.fillna("")
    partdic = {}
    for index, row in exam.iterrows():
        Partlist = []
        parts = [[], [], []]
        if row["三级项目"] is not np.nan:
            parts[2] = row['三级项目'].split("|")
            Partlist.append(parts[2][0])

        if row["二级项目"] is not np.nan:
            parts[1] = row['二级项目'].split("|")
            Partlist.append(parts[1][0])

        if row["一级项目"] is not np.nan:
            parts[0] = row['一级项目'].split("|")
            Partlist.append(parts[0][0])

        for i in range(3):
            newlist = Partlist[::-1]
            newlist = newlist[:i+1]
            if parts[i] != []:
                partdic[tuple(newlist)] = parts[i]
    return partdic

knowledgegraph={}
exam_special = {}
for m in modalities:
    knowledgegraph[m] = read_knowledgegraph(m)
    exam_special[m] = Get_special_exam(m)



#读取参数文件
conf = configparser.ConfigParser()
conf.read('knowledge_graph\\system_config.ini')
stop_pattern = conf.get("sentence", "stop_pattern")
sentence_pattern = conf.get("sentence", "sentence_pattern")
tipwords = conf.get("sentence", "tipwords")
punctuation = conf.get("clean", "punctuation")
ignore_keywords = conf.get("clean", "ignore_keywords")
Ignore_sentence = conf.get("clean", "Ignore_sentence")
stopwords = conf.get("clean", "stopwords")
second_root = conf.get("clean", "second_root")
spine_words = conf.get("clean", "spine")
dualparts=conf.get("orientation", "dualparts")
Exam_position = conf.get("Part_standard", "Exam_position")
Exam_enhance = conf.get("Part_standard", "Exam_enhance")
MaleKeyWords = conf.get("sex", "MaleKeyWords")
FemaleKeyWords = conf.get("sex", "FemaleKeyWords")
partlist_len={}
partlist_len['CT']=int(conf.get("Part_standard", "CT_partlist_len"))
partlist_len['MR']=int(conf.get("Part_standard", "MR_partlist_len"))
partlist_len['DX']=int(conf.get("Part_standard", "DX_partlist_len"))
partlist_len['MG']=int(conf.get("Part_standard", "MG_partlist_len"))
partlist_len_add=conf.get("Part_standard", "partlist_len_add").split("|")
def Interval_cross(a, b, extend=False):
    """判断两个区间是否相交"""
    if len(a) < 2 or len(b) < 2:
        return False
    if extend:
        a_start = a[0] % 100 if a[0] < 500 else a[0]
        a_end = a[1] % 100 if a[1] < 500 else a[1]
        b_start = b[0] % 100 if b[0] < 500 else b[0]
        b_end = b[1] % 100 if b[1] < 500 else b[1]
    else:
        a_start = a[0]
        a_end = a[1]
        b_start = b[0]
        b_end = b[1]
    if abs(a_start-b_start) > 100:
        return False
    if a_start < b_end:
        return a_end > b_start
    else:
        return b_end > a_start

def drop_dict_duplicates(list_of_dicts, keys):
    """去除字典列表的重复项"""
    seen = set()
    result = []
    for d in list_of_dicts:
        dict_key = tuple(d[key] for key in keys)
        if dict_key not in seen:
            seen.add(dict_key)
            result.append(d)
    return result

def GetPartTable(pre_ReportStr, keywords, stops):  
    """根据关键词填补句子的其他信息"""
    result = []
    i = 0
    for i in range(len(keywords)):
        k = keywords[i]
        info = {}
        start = 0
        end = len(pre_ReportStr)
        info['partlist'] = k[0][:-1]
        info['root'] = info['partlist'][0]
        info['axis'] = k[0][-1]
        temp = [x for x in stops if x <= k[1]]
        if temp != []:
            start = temp[-1]+1
        temp = [x for x in stops if x >= k[2]]
        if temp != []:
            end = temp[0]
            info['primary'] = pre_ReportStr[start:end].replace(" ", "")
        else:
            info['primary'] = pre_ReportStr[start:].replace(" ", "")
        # global ignore_keywords
        # info['ignore'] = any(info['primary'].startswith(element)
        #                     for element in ignore_keywords)
        info['ignore'] = starts_with_ignore(info['primary'])
        # print('primary=',info['primary'],"start=",start,"end=",end)
        info['words'] = pre_ReportStr[k[1]:k[2]]
        info['start'] = start
        info['word_start'] = k[1]
        info['word_end'] = k[2]
        info['sentence_end'] = end
        info['orientation'] = ''
        info['ambiguity'] = False
        search_str = pre_ReportStr[start:k[1]]
        # 从后往前搜索方位词
        mat = re.search("[左|右|两|双]", search_str[::-1])
        if mat:
            info['orientation'] = mat.group(0)
            if info['orientation'] == "两":
                info['orientation'] = "双"
        else:
            if info['words']=="左心":
                info['orientation'] = "左"
            if info['words']=="右心":
                info['orientation'] = "右"
        result.append(info)
    return result

def starts_with_ignore(string):
    """判断是否为忽略句"""
    global Ignore_sentence
    if re.search(Ignore_sentence, string,re.I):
        return True
    else:
        return False

def padding_sentence(dict_list, pre_ReportStr, stops):
    """扩展完整句子"""
    if len(dict_list) == 1:
        if dict_list[0]["sentence_end"] < stops[-1]:
            if not starts_with_ignore(pre_ReportStr[dict_list[0]["sentence_end"]+1:stops[-1]]):
                dict_list[0]["sentence_end"] = int(stops[-1])
                dict_list[0]["primary"] = pre_ReportStr[dict_list[0]
                                                        ['start']:stops[-1]].replace(" ", "")
        # Convert the list of dictionaries back to a dataframe and return it
        return dict_list

    # Get a list of unique sentence start and end positions, and sort them
    sentence_list = list(
        set([d['start'] for d in dict_list] + [d['sentence_end'] for d in dict_list]))
    sentence_list.sort()

    # If there are at least two sentences
    if len(sentence_list) >= 2:
        n = 1
        # Loop through each sentence pair
        while n <= len(sentence_list)-1:
            # If it's the last sentence pair, only consider stops after the last sentence end
            if n == len(sentence_list)-1:
                stoplist = [x for x in stops if x > sentence_list[n]]
            # Otherwise, consider stops between the current and next sentence
            else:
                stoplist = [x for x in stops if x >
                            sentence_list[n] and x < sentence_list[n+1]]

            # If there are stops between the sentences and the current sentence doesn't start with ignore,
            # loop through each stop and update the sentence end and primary columns
            if stoplist != [] and (pre_ReportStr[sentence_list[n]] not in "。;；？\n\r"):
                for x in stoplist:
                    if starts_with_ignore(pre_ReportStr[sentence_list[n]+1:x]):
                        break
                    for d in dict_list:
                        if d["start"] == sentence_list[n-1]:
                            d["sentence_end"] = int(x)
                            d["primary"] = pre_ReportStr[sentence_list[n-1]:x].replace(" ", "")
                    if pre_ReportStr[x] in "。;；？\n\r":
                        break
            n += 2

    # Convert the list of dictionaries back to a dataframe and return it
    return dict_list

def merge_part(data_dict, pre_ReportStr):
    """合并句子中在同一枝条的实体"""
    if data_dict == []:
        return []
    data_dict = [dict(t)
                 for t in {tuple(d.items()) for d in data_dict}]
    sentence_start = set(row['start'] for row in data_dict)

    # Create an empty list to store the processed data
    data = []

    # Loop through each sentence start position
    for start in sentence_start:
        # Get all rows with the current start position
        sentence = [row for row in data_dict if row['start'] == start]

        # Add a new key 'merge' to each row and set all values to False
        for row in sentence:
            row['merge'] = False

        # Remove the last character of the 'primary' field if it's a period or comma
        if sentence[0]['primary'][-1] == '.' or sentence[0]['primary'][-1] == ',':
            sentence[0]['primary'] = sentence[0]['primary'][:-1]

        # If there's only one row, append it to the processed data and continue to the next start position
        if len(sentence) == 1:
            data.append(sentence[0])
            continue

        sentence.sort(key=lambda x: x['partlist_length'])
        for m in range(len(sentence)-1):
            if sentence[m]['merge'] == True:
                continue
            for n in range(m+1, len(sentence)):
                punctStr = pre_ReportStr[sentence[m]
                                            ['word_end']:sentence[n]['word_start']]
                if re.search("[\\|/|+]", punctStr):
                    if (set(sentence[m]['partlist']) >= set(sentence[n]['partlist']) and
                            sentence[m]['orientation'] == sentence[n]['orientation']):
                        sentence[n]['merge'] = True
                else:
                    if (set(sentence[m]['partlist']) <= set(sentence[n]['partlist']) and
                            sentence[m]['orientation'] == sentence[n]['orientation']):
                        sentence[m]['merge'] = True
                        break

        # Append the rows with 'merge' set to False to the processed data
        for row in sentence:
            if row['merge'] == False:
                data.append(row)

    # Convert the processed data to a DataFrame, sort it by 'start' and 'word_start' columns, and reset the index
    data = sorted(data, key=lambda x: (x['start'], x['word_start']))
    # df_process = df_process.reset_index(drop=True)

    # Return the processed DataFrame
    return data

def get_ambiguity(sentence) -> tuple:
    """获取歧义词，并返回歧义词列表，单字列表，歧义词列表”“”"""

    ambiguity_sentence = []
    solo_sentence = []
    word_end = sentence[0]['word_end']

    ambiguity_sentence = [s for s in sentence if s["word_start"] < word_end ]
    sentence = [s for s in sentence if s not in ambiguity_sentence]

    longest_word = max(len(word['words']) for word in ambiguity_sentence)
    ambiguity_sentence = [word for word in ambiguity_sentence if len(
        word['words']) == longest_word]
    ambiguity_sentence = drop_dict_duplicates(ambiguity_sentence, ['partlist'])

    if len(ambiguity_sentence) == 1:
        solo_sentence = ambiguity_sentence
        ambiguity_sentence = []
    else:
        for word in ambiguity_sentence:
            word['ambiguity'] = True

    return sentence, solo_sentence, ambiguity_sentence
def xmerge(a, b):
    """将两个列表穿插组合."""
    alen, blen = len(a), len(b)
    mlen = max(alen, blen)
    for i in range(mlen):
        if i < alen:
            yield a[i]
        if i < blen:
            yield b[i]

def clean_mean(data_dict, pre_ReportStr):
    "判断歧义，并消解歧义"
    for i, d in enumerate(data_dict):
        partlist_length = len(d['partlist'])
        data_dict[i]['partlist_length'] = partlist_length
        data_dict[i]['index'] = i
    global punctuation
    df_process = []
    ambiguity_list = []
    sentence_start = set([d['start'] for d in data_dict])
    # Loop over sentence_start list

    for start in sentence_start:
        # Get all rows with the current start position
        sentence = [row for row in data_dict if row['start'] == start]
        sentence.sort(key=lambda x: x['word_start'], reverse=False)
        if len(sentence) == 1:
            df_process += sentence
            continue

        while len(sentence) > 0:
            sentence, solo_sentence, ambiguity_sentence = get_ambiguity(
                sentence)
            df_process += solo_sentence
            if len(ambiguity_sentence) > 0:
                ambiguity_list.append(ambiguity_sentence)
    clean_sentence = clean_ambiguity_sentence(df_process, ambiguity_list)
    return clean_sentence

def clean_ambiguity_sentence(process_list, ambiguity_list):
    """逐句消解歧义"""
    process_list = sorted(process_list, key=lambda x: x['index'])

    # Create an empty list to store the disambiguated parts
    clean_sentence = []

    # Iterate over each item in the ambiguity list
    for ambiguity in ambiguity_list:
        ambiguity_find = False
        #处理特殊情况
        if re.search(spine_words,ambiguity[0]['primary']):
            ambiguity=[x for x in ambiguity if "女性附件" not in x['partlist']]
        temp=[]        
        # Find the indices of parts that come before and after the current ambiguity            
        #参考上下文来确定语义
        previous_list = [x['index']
                        for x in process_list if x['index'] < ambiguity[0]['index']]
        next_list = [x['index']
                    for x in process_list if x['index'] > ambiguity[-1]['index']]
        search_list = [i for i in xmerge(previous_list[::-1], next_list)]
        # Iterate over each adjacent part and try to disambiguate the current part
        for n in search_list:
            adjacentPart = [p for p in process_list if p['index'] == n][0]
            if re.search(spine_words,adjacentPart['primary']):
                ambiguity=[x for x in ambiguity if "女性附件" not in x['partlist']]
            temp = [x for x in ambiguity if x['position'] in adjacentPart['partlist'] or 
                                            adjacentPart['position'] in x['partlist']]
            if len(temp) == 0 or len(temp)>1:
                adjacent = [
                    p for p in process_list if p['index'] == n][0]['axis']
                temp = [x for x in ambiguity if Interval_cross(
                    x['axis'], adjacent)]
            if len(temp) > 0 and len(temp)<len(ambiguity):
                clean_sentence.extend(temp)
                ambiguity_find = True
                break
        
        # 如果以上没有匹配，则扩展坐标范围再次匹配，把相邻的纵轴柱纳入匹配
        if not ambiguity_find:
            for n in search_list:
                adjacentPart = [
                    p for p in process_list if p['index'] == n][0]['partlist']
                temp = [x for x in ambiguity if Interval_cross(
                    x['axis'], adjacent, extend=True)]
                if len(temp) > 0:
                    clean_sentence.extend(temp)
                    ambiguity_find = True
                    break

        # 若没有上下文/附加信息可以消解歧义的处理
        if not ambiguity_find:
            # 优先部位作为根节点
            priority = [x for x in ambiguity if re.search(
                second_root, x['position'])]
            if len(priority) > 0:
                ambiguity[0]['partlist'] = tuple([priority[0]['position']])
                ambiguity[0]['root'] = priority[0]['position']
                ambiguity[0]['position'] = priority[0]['position']
                clean_sentence.append(ambiguity[0])
            else:
                ambiguity = [x for x in ambiguity if x['partlist_length'] > 1]
                # 同时属于多个二级节点的部位，全部保留
                second_part = [x['partlist'][1] for x in ambiguity]
                if len(set(second_part)) == 1:
                    clean_sentence.extend(ambiguity)
                else:
                    # 分词不同的部位，保留位置靠前，word_start较小的
                    start_last = max([x['word_start'] for x in ambiguity])
                    start_first = min([x['word_start'] for x in ambiguity])
                    if start_last > start_first:
                        ambiguity = [
                            x for x in ambiguity if x['word_start'] == start_first]
                        clean_sentence.append(ambiguity[0])
                    else:
                        clean_sentence.extend(ambiguity)
    process_list.extend( clean_sentence)
    return process_list

def fill_orientation(data_dict,pre_ReportStr):
    """"填充无方向的部位"""
    global sentence_pattern
    sentence_end = [match.start()
                for match in re.finditer(sentence_pattern, pre_ReportStr)]
    for i, part in enumerate(data_dict):
        if part['orientation']=='' and re.search(dualparts," ".join(part['partlist'])):
            previous_end=[x for x in sentence_end if x<part['start']]
            if previous_end==[]:
                break
            for o in data_dict[i::-1]:
                # if o['sentence_end']<=previous_end[-1]:
                #     break
                if o['orientation']!="":
                    data_dict[i]['orientation']=o['orientation']
                    break
    return data_dict

def get_base_info(ReportStr: str,modality:str):
    """实体抽取主函数."""

    #脊柱简写预处理
    if type(ReportStr)!=str or len(ReportStr)==0:
        return []
    pre_ReportStr = Str_replace(ReportStr, modality)

    global stop_pattern
    result = []
    if len(pre_ReportStr) == 0:
        return result
    if not (re.search(stop_pattern, pre_ReportStr[-1])):
        pre_ReportStr = pre_ReportStr+'\n'
    stops = [match.start()
             for match in re.finditer(stop_pattern, pre_ReportStr)]
    # sentences=re.split(stop_pattern,ReportStr)
    for kg in knowledgegraph[modality] :
        BodyProcessor=None
        BodyProcessor = KeywordProcessor()
        BodyProcessor.add_keywords_from_dict(kg)
        keywords = BodyProcessor.extract_keywords(
            pre_ReportStr, span_info=True)
        if keywords != []:
            temp = GetPartTable(pre_ReportStr, keywords, stops)
            result += temp
    if len(result) == 0:
        return []
    result = sorted(result, key=lambda x: (x['start'], x['word_start']))
    for i, d in enumerate(result):
        result[i]['position'] = d['partlist'][-1]
    result = padding_sentence(result, pre_ReportStr, stops)
    result = clean_mean(result, pre_ReportStr)
    result = merge_part(result, pre_ReportStr)

    #填补缺失的方向信息
    for d in result:
        if d['orientation'] != '':
            continue
        if re.search("左(?!.{0,3}位)",d['primary']) and d['orientation'] == '':
            d['orientation'] = '左'
        if re.search("右(?!.{0,3}位)",d['primary']) and d['orientation'] == '':
            d['orientation'] = '右'
        if re.search("双(?!.{0,3}位)",d['primary']) and d['orientation'] == '':
            d['orientation'] = '双'
    result=fill_orientation(result,pre_ReportStr) 
    #清洗结果并输出            
    result = [{k: v for k, v in d.items() if k not in ["word_end",
                                                       "sentence_end", "partlist_length", "merge"]} for d in result]


    return result


def get_study_info(studypart:str, modality:str, debug=False):
    if debug:
        start_time = time.time()
    """解析检查部位获得标准化的部位列表，包括部位分类、检查方式、患者方位、检查方位、特殊检查."""
    modality=modality.upper()
    studypart_analyze = get_base_info(studypart, modality) if studypart else []
    if len(studypart_analyze) == 0:
        return []
    result=[]
    for part in studypart_analyze:
        #检查体位
        part['Exam_position'] = re.findall(Exam_position, part['primary'])
        part['Exam_position'] = list(set(part['Exam_position']))
        if modality in ['DX','MG'] and part['Exam_position']==[]:
            part['Exam_position']=['正位']
        #增强方式
        part['Exam_method'] =[]
        if modality in ['CT','MR']:
            if "平扫" in part['primary']:
                part['Exam_method'].append('平扫')
            if re.search(Exam_enhance, part['primary']) or (modality=='CT' and part['position']=="脑动脉"):
                part['Exam_method'].append('增强')
            if part['Exam_method']==[]:
                part['Exam_method']=["平扫"]
            if part['Exam_method']==['增强'] and re.search("血管|动脉",' '.join(part['partlist'])) is None and "直接增强" not in part['primary']:
                part['Exam_method'].append("平扫")
        #特殊检查
        part['Exam_special'] = ''
        Special_Processor = KeywordProcessor()
        Special_Processor.add_keywords_from_dict(exam_special[modality])
        part['Exam_special'] = list(
            set(Special_Processor.extract_keywords(part['primary'])))
        result.append(part)
    if debug:
        end_time = time.time()
        print("analysis time=%.3f秒" % (end_time-start_time))
    return result

def shorten_partlist(StudyPart,modality):
    "剪枝，缩短partlist到指定长度"
    global partlist_len,partlist_len_add
    for d in StudyPart:
        if d['root'] in partlist_len_add:
            d['position']=d['partlist'][partlist_len[modality]] if len(d['partlist'])>partlist_len[modality] else d['partlist'][-1]
        else:
            d['position']=d['partlist'][partlist_len[modality]-1] if len(d['partlist'])>partlist_len[modality]-1 else d['partlist'][-1]
    return StudyPart

def judge_sex(StudyPart):
    "判断部位的性别"
    global MaleKeyWords,FemaleKeyWords
    if re.search(MaleKeyWords," ".join(StudyPart['partlist'])):
        return "M"
    elif re.search(FemaleKeyWords," ".join(StudyPart['partlist'])):
        return "F"
    else:
        return ""
    
def check_match(StudyPartStr: str, HistoryPartStr: str,modality:str):
    # 输入格式
    # 当前检查部位：StudyPartStr="肝胆胰脾"
    # 其他检查名称列表：HistoryPart='上腹部、胸部/纵隔'
    # 返回格式：如果部位重叠，返回共同的标准化根节点，如果不重叠，返回[]
    match_root=[]
    modality=modality.upper()
    StudyPart = get_study_info(StudyPartStr, modality)
    if len(StudyPart) == 0:
        return match_root
    his_position = get_study_info( HistoryPartStr, modality)
    if len(his_position) == 0:
        return match_root
    #调整partlist长度,决定在哪个层级进行比较
    StudyPart=shorten_partlist(StudyPart,modality)
    his_position=shorten_partlist(his_position,modality)
    for row in StudyPart:
        #判断部位和方向是否相符
        temp = [x for x in his_position if ((x['position'] in row['partlist']) or 
            (row['position'] in x['partlist'])) and ((x['orientation'] == row['orientation']) or
                 ('双' in x['orientation']) or ('双' in row['orientation']) or
                 (x['orientation'] == '') or (row['orientation'] == ''))]
        if len(temp)==0:
            continue
        #判断性别是否相符
        s_sex=judge_sex(row)
        h_sex=list(map(judge_sex,temp))
        h_sex=[x for x in h_sex if x!='']
        if s_sex and h_sex:
            if s_sex!=h_sex[0]:
                break
        #判断增强方式是否相符
        Exam_methods=[]
        [Exam_methods.extend(x['Exam_method']) for x in temp]
        if modality in ['CT','MR'] and '头颅血管' not in row['partlist']:
            temp = [x for x in temp if (set(Exam_methods).intersection(set(row['Exam_method'])) or (Exam_methods==row['Exam_method'])) ]
        if modality in ['DX','MG']:
            #判断检查体位是否相符
            Exam_positions=[]
            [Exam_positions.extend(x['Exam_position']) for x in temp]
            temp = [x for x in temp if (set(Exam_positions).intersection(set(row['Exam_position'])) or (Exam_positions==row['Exam_position']))]
            #判断特殊检查是否相符
            Exam_specials=[]
            [Exam_specials.extend(x['Exam_special']) for x in temp]
            temp = [x for x in temp if (set(Exam_specials).intersection(set(row['Exam_special'])) or (Exam_specials==row['Exam_special']))]
        if len(temp) > 0:
            match_root.append(row['root'])
    return list(set(match_root))

#读取广东省互认目录
def get_Guandong_dict(filename):
    GuangdongHR=pd.read_excel(filename)
    result=[]
    for index,row in GuangdongHR.iterrows():
        item=get_study_info(row['项目中文简称'], row['检查模态'])
        if len(item)>0:
            item = [{**dic, 'modality': row['检查模态']} for dic in item]
            item = [{**dic, 'code': row['序号']} for dic in item]
            item = [{**dic, 'name': row['项目中文简称']} for dic in item]
            result.extend(item)
    return result
Guangdong_dict=get_Guandong_dict("knowledge_graph\\GuangdongHR.xlsx")
def Generate_GuandongHR_code(StudyPartStr:str,modality:str,output='code',debug=False):
    if debug:
        start_time = time.time()
    modality=modality.upper()
    StudyPart = get_study_info(StudyPartStr, modality)
    if len(StudyPart) == 0:
        return []
    HR_dict=[x for x in Guangdong_dict if x['modality']==modality]
    if len(HR_dict)==0:
        return []
    result=[]
    #决定在哪个层级进行比较
    StudyPart=shorten_partlist(StudyPart,modality)
    HR_dict=shorten_partlist(HR_dict,modality)
    for row in StudyPart:
        #判断部位和方向是否相符
        # position=row['partlist'][1] if len(row['partlist'])>1 else row['partlist'][0]
        temp = [x for x in HR_dict if (row['position'] in x['partlist']) and 
                ((x['orientation'] == row['orientation']) or
                ('双' in x['orientation']) or ('双' in row['orientation']) or
                (x['orientation'] == '') or (row['orientation'] == ''))]
        if len(temp)==0:
            continue
        #判断增强方式是否相符
        temp = [x for x in temp if set(x['Exam_method']).issubset(set(row['Exam_method'])) ]
        #判断检查体位是否相符
        temp = [x for x in temp if set(x['Exam_position']).issubset(set(row['Exam_position']))]
        #判断特殊检查是否相符
        if modality in ['DX','MG']:
            temp = [x for x in temp if set(x['Exam_special']).issubset(set(row['Exam_special'])) ]
        if len(temp) > 0:
            if output=='code':
                result.extend([x['code'] for x in temp])
            if output=='name':
                result.extend([ f"{x['code']}_{x['name']}" for x in temp])
    if debug:
        print("analysis time=%.3f秒" % (time.time()-start_time))
    return list(set(result))



if __name__ == "__main__":
    ReportStr = """三叉神经/面神经3.0T"""
    # print(Generate_GuandongHR_code(ReportStr,  "DX",output='name' ,debug=True))
    result=get_study_info(ReportStr, "MR",debug=True)
    # pprint([[x['orientation'],x['partlist'],x['Exam_method'],x['Exam_position'],x['Exam_special']] for x in result])
    print(result)