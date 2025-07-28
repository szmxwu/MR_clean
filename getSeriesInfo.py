import requests
import json
import pandas as pd
import numpy as np
from preprocess import correct_multi_part_studies
from MR_clean import process_mri_dataframe
from interAccreditation import get_study_info,drop_dict_duplicates
from time import time
url="http://172.17.250.136:5000/api/mr/meta/"
import pyodbc
server = '172.17.250.190\\CSSERVER'
database = 'GCRIS2'
username = 'RIS'
password = 'RIS'
driver= '{ODBC Driver 17 for SQL Server}' # 确保使用了正确的驱动程序
engine=pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)

def get_chekingItems(accno):
    sqlstr="""
        select  tRegorder.AccNo as 影像号,tRegProcedure.CheckingItem as 部位 
        from tRegorder,tRegProcedure 
        where tRegorder.OrderGuid=tRegProcedure.OrderGuid and 
        tRegorder.AccNo='%s'""" %accno
    risQueryResult= pd.read_sql(sqlstr, engine)
    if len(risQueryResult)==0:
        return ''
    return ','.join(risQueryResult['部位'].tolist())
def prepare_df(df):
    if len(df)==0:
        return pd.DataFrame
    df['study_info']=None
    df['standardPart']=None
    df['Position_orientation'] =None
    df['Exam_orientation']  =None
    df['Exam_special']  =None
    df['Exam_method'] =None
    df['position'] =None
    df['parts_sum'] =np.nan
    df['checkingitem']=get_chekingItems(df.at[0,'AccessionNumber'])
    stardData = get_study_info(df.at[0, 'checkingitem'],df.at[0, 'Modality'])
    stardData=drop_dict_duplicates(stardData, ['root'])
    if len(stardData)==0:
        return pd.DataFrame
    parts=[x['root'] for x in  stardData]
    df['root'] =",".join(parts)
    if len(parts)>1:
        df['study_info'] = str(stardData)
        df['parts_sum'] =len(parts)
    else:
        df['Position_orientation'] = ",".join([x['orientation'] for x in  stardData])
        df['Exam_orientation'] = ",".join([",".join(x['Exam_position']) for x in stardData])
        df['Exam_special'] = ",".join(["-".join(x) for x in stardData[0]['Exam_special']])
        df['Exam_method'] = ",".join([",".join(x['Exam_method']) for x in stardData])
        df['position'] = stardData[0]['position']
        df['standardPart']  =",".join(parts)
        df['parts_sum'] =1
    return df

def getInof(accno):
    response=requests.get(url+accno)
    result=json.loads(response.text)
    result=pd.DataFrame(result['series_data'])
    stard_result=prepare_df(result)
    stard_result=pd.DataFrame(stard_result)
    stard_result = correct_multi_part_studies(stard_result)
    stard_result=process_mri_dataframe(stard_result)
    stard_result['SeriesTime'] = stard_result['SeriesTime'].astype(str).str.zfill(6)  # 将 "81955" 转换为 "081955"
    # 拼接 SeriesDate 和 SeriesTime 字段
    stard_result['StudyDateTime'] = pd.to_datetime(
        stard_result['SeriesDate'].astype(str) + stard_result['SeriesTime'].astype(str),
        format='%Y%m%d%H%M%S'
    )
    stard_result.sort_values(by='StudyDateTime', ascending=True, inplace=True)
    selectCol=["SeriesNumber",'StudyDateTime','checkingitem'	,'standardPart',	'parts_sum',	'Exam_orientation',	'standardFieldStrength'	,'standardManufacturer',	'cleanedModelName',	'sequenceClass'	,'protocolName_lower',	'standardOrientation'	,'standardDimension'	,'isFatSuppressed',	'isContrastEnhanced', 'dynamicPhase',	'hasMotionCorrection']
    if stard_result['Exam_orientation'].isna().sum()==0:
        selectCol.remove("Exam_orientation")
    if stard_result.at[0,'parts_sum']==1:
        selectCol.remove("parts_sum")
    stard_result= stard_result[selectCol]
    stard_result.rename(columns={
        "SeriesNumber":"序号",
        "StudyDateTime":"检查日期",
        "checkingitem":"原始项目名",
        "standardPart":"标准部位",	
        "parts_sum":"部位数量",	
        "Exam_orientation":"部位方位",
        "standardFieldStrength":"场强"	,
        "standardManufacturer":"制造商",	
        "cleanedModelName":"型号",		
        "protocolName_lower":"原序列名",
        "standardOrientation":"图像方位",
        "standardDimension":"图像维度"	,
        "isFatSuppressed":"压脂",	
        "isContrastEnhanced":"增强", 
        "dynamicPhase":"动态",	
        "hasMotionCorrection":"运动校正",	
        "sequenceClass":"标准序列"
    },inplace=True,errors='ignore')
    stard_result=stard_result.replace({True: "√", False: ""})
    return stard_result
if __name__=="__main__":
    accno="M25061700436"
    accno="Z25070301862"
    start=time()
    stard_result=getInof(accno)
    html_table = stard_result.to_html(index=False,classes="table table-striped")
    with open('dataframe_output.html', 'w', encoding='utf-8') as f:
        f.write(html_table)
    print(f"消耗时间{(time()-start):.1f}秒")
