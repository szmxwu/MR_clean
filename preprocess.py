import pandas as pd
import numpy as np
import re
import os
import ast
from sklearn.cluster import KMeans
import warnings
from interAccreditation import get_study_info,drop_dict_duplicates, knowledgegraph
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 导入您配置好的先验知识字典
try:
    from anatomical_priors_config import ANATOMICAL_PRIORS
except ImportError:
    print("错误: 无法找到 'anatomical_priors_config.py' 文件。")
    print("请确保该配置文件与 clean.py 在同一个文件夹下。")
    ANATOMICAL_PRIORS = {} # 定义一个空字典以避免程序崩溃

# 忽略KMeans关于n_init的未来警告，保持输出整洁
warnings.filterwarnings("ignore")

def prepare_data(df):
    df['study_info']=None
    df['standardPart']=None
    df['Position_orientation'] =None
    df['Exam_orientation']  =None
    df['Exam_special']  =None
    df['Exam_method'] =None
    df['position'] =None
    df['parts_sum'] =np.nan
    for index in tqdm(range(len(df)), desc="预处理数据"):
        stardData = get_study_info(df.at[index, 'checkingitem'],df.at[index, 'Modality'])
        stardData=drop_dict_duplicates(stardData, ['root'])
        if len(stardData)==0:
            continue
        parts=[x['root'] for x in  stardData]
        df.iloc[index, 'root'] =",".join(parts)
        if len(parts)>1:
            df.iloc[index, 'study_info'] = str(stardData)
            df.iloc[index, 'parts_sum'] =len(parts)
        else:
            df.iloc[index, 'Position_orientation'] = ",".join([x['orientation'] for x in  stardData])
            df.iloc[index, 'Exam_orientation'] = ",".join([",".join(x['Exam_position']) for x in stardData])
            df.iloc[index, 'Exam_special'] = ",".join(["-".join(x) for x in stardData[0]['Exam_special']])
            df.iloc[index, 'Exam_method'] = ",".join([",".join(x['Exam_method']) for x in stardData])
            df.iloc[index, 'position'] = stardData[0]['position']
            df.iloc[index, 'standardPart']  =",".join(parts)
            df.iloc[index, 'parts_sum'] =1
    return df

def process_row(args):
    """
    处理单行数据的函数
    Args:
        args: 元组 (index, row, modality_col, checkingitem_col)
    Returns:
        处理后的行数据 (Series)
    """
    index, row, modality_col, checkingitem_col = args
    # 创建行的副本以避免修改原始数据
    row = row.copy()
    
    # 初始化新列
    row['study_info'] = None
    row['standardPart'] = None
    row['Position_orientation'] = None
    row['Exam_orientation'] = None
    row['Exam_special'] = None
    row['Exam_method'] = None
    row['position'] = None
    row['parts_sum'] = np.nan
    
    # 处理单行数据
    stardData = get_study_info(row[checkingitem_col], row[modality_col])
    stardData = drop_dict_duplicates(stardData, ['root'])
    
    if len(stardData) > 0:
        parts = [x['root'] for x in stardData]
        row['root'] = ",".join(parts)
        
        if len(parts) > 1:
            row['study_info'] = str(stardData)
            row['parts_sum'] = len(parts)
        else:
            row['Position_orientation'] = ",".join([x['orientation'] for x in stardData])
            row['Exam_orientation'] = ",".join([",".join(x['Exam_position']) for x in stardData])
            row['Exam_special'] = ",".join(["-".join(x) for x in stardData[0]['Exam_special']])
            row['Exam_method'] = ",".join([",".join(x['Exam_method']) for x in stardData])
            row['position'] = stardData[0]['position']
            row['standardPart'] = ",".join(parts)
            row['parts_sum'] = 1
    
    return row

def prepare_data_parallel(df, modality_col='Modality', checkingitem_col='checkingitem', num_processes=None):
    """
    并行处理DataFrame的多进程版本（基于行处理）
    Args:
        df: 输入的DataFrame
        modality_col: 模态列名
        checkingitem_col: 检查项目列名
        num_processes: 进程数，默认为CPU核心数
    """
    # 设置进程数
    if num_processes is None:
        num_processes = min(cpu_count(), 8)  # 限制最大进程数，避免内存问题
    
    # 准备参数：每行数据 + 索引 + 列名
    args_list = [(i, row, modality_col, checkingitem_col) 
                for i, row in df.iterrows()]
    
    # 使用进程池处理
    with Pool(processes=num_processes) as pool:
        # 使用tqdm显示每条记录的进度
        results = list(tqdm(pool.imap(process_row, args_list, chunksize=50), 
                      total=len(df),
                      desc="处理记录"))
    
    # 重新组装DataFrame
    processed_df = pd.DataFrame(results)
    
    # 恢复原始索引
    processed_df.index = df.index
    
    return processed_df


def correct_multi_part_studies(df):
    """
    利用先验知识字典，通过空间聚类修正多部位研究中每个序列的部位归属。

    Args:
        df (pd.DataFrame): 经过初步标准化的DataFrame。

    Returns:
        pd.DataFrame: 增加了'correctedPart'列的DataFrame。
    """
    print("\n开始执行多部位研究的部位归属修正...")
    
    # 1. 初始化新列，默认等于原始的标准化部位
    df['standardPart']=df['root']

    # 2. 识别所有需要处理的多部位研究
    multi_part_mask = df['standardPart'].str.contains(',', na=False)
    studies_to_process = df[multi_part_mask]['StudyInstanceUID'].unique()
    
    if len(studies_to_process) == 0:
        print("未发现需要修正的多部位研究。")
        return df

    print(f"发现 {len(studies_to_process)} 个多部位研究需要进行序列级归属修正。")

    # 3. 准备数据，安全地解析IPP向量
    def parse_vector(val):
        try:
            vector = ast.literal_eval(str(val))
            if isinstance(vector, list) and len(vector) == 3:
                return [float(v) for v in vector]
        except (ValueError, SyntaxError, TypeError):
            return None
    
    df['ipp_vector'] = df['ImagePositionPatient'].apply(parse_vector)
    
    # 4. 逐个研究进行处理
    processed_count = 0
    for study_uid in studies_to_process:
        study_df = df[df['StudyInstanceUID'] == study_uid].copy()
        
        # 获取该研究的关键信息
        part_string = study_df['standardPart'].iloc[0]
        patient_position = str(study_df['PatientPosition'].iloc[0])
        if part_string is None or patient_position is None:
            print(f"  - 警告: 研究 {study_uid} 中缺少必要信息。跳过。")
            continue
        # 检查先验知识字典中是否存在对应的规则
        k=len(part_string.split(','))
        axis='z'
        part_dict=[]
        for part in part_string.split(','):
            for dictionary in knowledgegraph['MR']:
                find=False
                temp={}
                for key in dictionary.keys():
                    if key[0] == part:  # 检查元组的第一个元素
                        temp['name']=key[0]
                        temp['axis']= key[-1][0]
                        part_dict.append(temp)
                        find=True
                        break
                if find: break
        part_dict = sorted(part_dict, key=lambda x: x.get("axis", float('inf')))
        if part_string  in ANATOMICAL_PRIORS:
            # print(f"  - 警告: 在 anatomical_priors_config.py 中未找到对 '{part_string}' 的配置。跳过研究 {study_uid}。")
            # continue

            # 从配置中获取规则
            priors = ANATOMICAL_PRIORS[part_string]
            k = len(priors["clusters"])
            axis = priors["distinguishing_axis"].lower()
        
        # 准备聚类数据
        study_df=study_df[study_df['ipp_vector'].notnull()]
        clustering_data = pd.DataFrame(study_df['ipp_vector'].tolist(), index=study_df.index, columns=['x', 'y', 'z'])
        
        if clustering_data.isnull().any().any():
            print(f"  - 警告: 研究 {study_uid} 中存在无效的空间位置数据。跳过。")
            continue
            
        # 执行K-Means聚类
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(clustering_data[[axis]])
        
        # 将聚类结果映射回原始DataFrame的行
        for i, original_index in enumerate(study_df.index):
            assigned_cluster_id = cluster_labels[i]
            
            # 查找该聚类ID对应的解剖学标签
            correct_part = "UNKNOWN"
            # 根据PatientPosition决定排序逻辑
            is_feet_first = 'FF' in patient_position.upper()
            
            # 在我们的配置中，簇列表已按区分轴的值从大到小排序
            # 头先进（HF）：Z值越大，越靠上，索引越小
            # 脚先进（FF）：Z值越大，越靠下，索引越小


            # 稳健的匹配方法：找到每个序列所属的簇，然后将该簇的平均坐标与配置中的坐标进行匹配
            cluster_centers = kmeans.cluster_centers_
            # 找到当前序列所属簇的中心坐标
            current_cluster_center = cluster_centers[assigned_cluster_id][0]
            if is_feet_first:
                sorted_list = sorted(cluster_centers)
            else:
                sorted_list = sorted(cluster_centers,reverse=True)
            index = sorted_list.index(current_cluster_center)
            correct_part=part_dict[index]['name']
            # 在配置中找到最接近的簇
            try:
                closest_cluster_config = min(priors["clusters"], key=lambda c: abs(c["distinguishing_axis_value"] - current_cluster_center))
                correct_part = closest_cluster_config["anatomical_label"]
            except:
                pass
            
            # 修正'standardPart'列的值
            df.loc[original_index, 'standardPart'] = correct_part
            study_info = ast.literal_eval(df.loc[original_index, 'study_info'])
            study_info=[x for x in study_info if x['root']==correct_part]
            df.loc[original_index,'Position_orientation'] = ",".join([x['orientation'] for x in study_info])
            df.loc[original_index,'Exam_orientation'] = ",".join([",".join(x['Exam_position']) for x in study_info])
            df.loc[original_index,'Exam_special'] = ",".join(["-".join(x) for x in study_info[0]['Exam_special']])
            df.loc[original_index,'Exam_method'] = ",".join([",".join(x['Exam_method']) for x in study_info])
            df.loc[original_index,'position'] = study_info[0]['position']

        processed_count += 1

    print(f"\n修正完成！共处理了 {processed_count} 个多部位研究。")
    
    # 清理临时列
    df.drop(columns=['ipp_vector'], inplace=True, errors='ignore')
    
    return df


if __name__ == '__main__':
    # combined_data = combine_csv_files("mr_meta_20250501_20250527-version6")
    # combined_data.to_csv('combined_data.csv', index=False)
    # df=pd.read_csv('combined_data.csv')
    # df=prepare_data_parallel(df, num_processes=5) 
    # df=prepare_data(df)
    df=pd.read_excel('combined_data.xlsx')
    df_processed = correct_multi_part_studies(df)
    df.to_excel('combined_data.xlsx', index=False)
    