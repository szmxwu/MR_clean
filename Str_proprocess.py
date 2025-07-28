import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
import warnings
import configparser
from pprint import pprint
warnings.filterwarnings("ignore")
#读取参数文件
conf = configparser.ConfigParser()
conf.read('knowledge_graph\\system_config.ini')
sentence_pattern = conf.get("sentence", "sentence_pattern")
spine_words = conf.get("clean", "spine")

modalities=['CT',"MR","DX","MG"]
PartReplaceTable={}
for m in modalities:
    PartReplaceTable[m] = pd.read_excel(
    'knowledge_graph\\preprocess.xlsx', sheet_name=m).to_dict('records')
ConditionReplaceTable = pd.read_excel(
    'knowledge_graph\\preprocess.xlsx', sheet_name='条件').to_dict('records')
#文本预处理：简写词还原配置
pattern1=re.compile(r'([^颈胸腰骶尾])(\d{1,2})[、|,|，|及|和](\d{1,2})([颈|胸|腰|骶|尾])(?!.*段)',flags=re.I)
pattern2=re.compile(r'([^颈胸腰骶尾/])(\d{1,2})[、|,|，|及|和]([颈|胸|腰|骶|尾])(\d{1,2})(?!.*段)' ,flags=re.I)
pattern3=re.compile(r'([^颈胸腰骶尾])(\d{1,2})(/\d{1,2})[、|,|，|及|和](\d{1,2})(/\d{1,2})([颈|胸|腰|骶|尾])' ,flags=re.I)
pattern4=re.compile(r'([^颈胸腰骶尾])(\d{1,2})(/\d{1,2})[、|,|，|及|和]([颈|胸|腰|骶|尾])(\d{1,2})(/\d{1,2})',flags=re.I) 
pattern5=re.compile(r'([腰|胸|颈|骶|尾|c|l|t|s])(\d{1,2})(/\d{1,2})?[、|,|，|及|和](\d{1,2})' ,flags=re.I)
pattern6=re.compile(r'([腰|胸|颈|骶|尾|c|l|t|s]\d{1,2})(/\d{1,2})?[,|，]([腰|胸|颈|骶|尾|c|l|t|s]\d{1,2})',flags=re.I) 
pattern7=re.compile(r"([胸|腰|颈|骶|c|t|l|s])(\d{1,2})/([胸|腰|颈|骶|c|t|l|s])?(\d{1,2})-([胸|腰|颈|骶|c|t|l|s])?(\d{1,2})/([胸|腰|颈|骶|c|t|l|s])?(\d{1,2})(?!.*段)",flags=re.I) 
pattern8=re.compile(r"([胸|腰|颈|骶|c|t|l|s])(\d{1,2})-([胸|腰|颈|骶|c|t|l|s])?(\d{1,2})(椎体)?(?!.*段)",flags=re.I) 
pattern9=re.compile(r'(^|[^a-zA-Z])C([1-8])(?!.*段)',flags=re.I) 
pattern10=re.compile(r'(^|[^a-zA-Z长短低高等脂水])T(\d{1,2})(?!.*[段_信号压黑为呈示a-zA-Z])',flags=re.I) 
pattern13=re.compile(r'(^|[^a-zA-Z])T(\d{1,2})椎',flags=re.I) 
pattern14=re.compile(r'(^|[^a-zA-Z])T([3-9]|10|11|12)(?!.*[MN])',flags=re.I) 
pattern11=re.compile(r'(^|[^a-zA-Z])S([1-5])(?!.*[段a-zA-Z0-9])',flags=re.I) 
pattern12=re.compile(r"([a-zA-Z\u4e00-\u9fa5])\.([a-zA-Z\u4e00-\u9fa5])",flags=re.I) 


def spine_extend(Str):
    """扩展椎体简写形式."""
    
    mat=pattern8.findall(Str)  
    if not mat:
        return Str
    try:
        for group in mat:
            new_str=""
            if group[0]==group[2] or group[2]=='':
                last=int(group[3])
            else:
                last=last_spine(group[0])
            start=int(group[1])
            end=int(group[3])
            for i in range(start,last+1):
                new_str+=group[0]+str(i)+"椎体、"
            if group[0]!=group[2] and group[2]!='':
                for i in range(1,end+1):
                    new_str+=group[2]+str(i)+"椎体、"
            old_str=group[0]+group[1]+"-"+group[2]+group[3]+group[4]
            Str=Str.replace(old_str,new_str[:-1])
    except:
        pass
    return Str

def last_spine(spine_str):
    if spine_str in "颈cC":
        last=7
    elif spine_str in "胸tT":
        last=12
    elif spine_str in "腰tT":
        last=5
    elif spine_str in "骶sS":
        last=5
    else:
        last=3
    return last

def disk_extend(Str):
    """扩展椎间盘简写形式."""
    mat=pattern7.findall(Str)  
    if not mat:
        return Str
    try:
        for group in mat:
            new_str=''
            if group[0]==group[4] or group[4]=='':
                last=int(group[5])
            else:
                last=last_spine(group[0])
            start=int(group[1])
            #下一段
            end=int(group[5])
            for i in range(start,last+1):
                new_str+=group[0]+str(i)+"/"+str(i+1)+"、"
            if group[0]!=group[4] and group[4]!='':
                for i in range(1,end+1):
                    new_str+=group[4]+str(i)+"/"+str(i+1)+"、"
            new_str=re.sub("[颈|c]7/8","颈7/胸1",new_str,flags=re.I)
            new_str=re.sub("[胸|t]12/13","胸12/腰1",new_str,flags=re.I)
            new_str=re.sub("[腰|l]5/6","腰5/骶1",new_str,flags=re.I)
            old_str=group[0]+group[1]+"/"+group[2]+group[3]+"-"+group[4]+group[5]+"/"+group[6]+group[7]
            Str=Str.replace(old_str,new_str[:-1])
    except:
        pass
    return Str

def extend_spine_dot(sentence):  
    """扩展椎体+顿号形式."""
    n=0
    sentence=pattern1.sub(r"\1\4\2、\4\3",sentence)
    sentence=pattern3.sub(r"\1\6\2\3、\6\4\5",sentence)
    while n<10:
        new_sentence=pattern2.sub(r"\1\3\2、\3\4",sentence)
        new_sentence=pattern4.sub(r"\1\4\2\3、\4\5\6",new_sentence)
        new_sentence=pattern5.sub(r"\1\2\3、\1\4",new_sentence)
        new_sentence=pattern6.sub(r"\1\2、\3",new_sentence)
        if new_sentence==sentence:
            break
        else:
            sentence=new_sentence
        n+=1
    return sentence
# %% 判断坐标区间相交
def Str_replace(ReportStr:str, modality:str):
    """用替换方式预处理."""
    if type(ReportStr) != str or len(ReportStr)==0:
        return ''
    ReportStr = re.sub("[ \xa0\x7f]", "",ReportStr)
    ReportStr = pattern12.sub(r"\1。\2", ReportStr)
    #还原缩写形式
    ReportStr=extend_spine_dot(spine_extend(disk_extend(ReportStr)))
    # 一般处理
    Replace_table= PartReplaceTable[modality] 
    for row in Replace_table:
        if row['原始值'] is np.nan:
            continue
        if row['替换值'] is np.nan:
            ReportStr = ReportStr.replace(row["原始值"], "")
        else:
            ReportStr = ReportStr.upper().replace(row["原始值"], row["替换值"])

    # 分句后处理
    sentence_end = [match.start()
                    for match in re.finditer(sentence_pattern, ReportStr)]
    if sentence_end == []:
        sentence_end = [len(ReportStr)]
    cstops = [0]
    cstops.extend(sentence_end)
    
    for i in range(len(cstops)-1):
        temp = ReportStr[cstops[i]:cstops[i+1]]
        if temp=="" :
            continue
        if re.search(spine_words, temp):
            temp = pattern9.sub('\\1颈\\2',temp)
            temp = pattern10.sub('\\1胸\\2',temp)
            temp = pattern13.sub('\\1胸\\2',temp)
            temp = pattern11.sub('\\1骶\\2',temp)
        temp = pattern14.sub('\\1胸\\2',temp)
        #正则替换
        for row in ConditionReplaceTable:
            if row['替换值'] is np.nan or row['原始值'] is np.nan:
                continue
            temp=re.sub(row['原始值'],row['替换值'],temp)
            ReportStr = ReportStr.replace(ReportStr[cstops[i]:cstops[i+1]], temp)
    return ReportStr