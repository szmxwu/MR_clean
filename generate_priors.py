import pandas as pd
import numpy as np
import ast
from sklearn.cluster import KMeans
import warnings

# 忽略KMeans关于n_init的未来警告，以保持输出整洁
warnings.filterwarnings("ignore")

def generate_anatomical_priors_dict(filepath):
    """
    读取标准化的MRI序列文件，通过空间聚类分析多部位研究，
    并生成一个可配置的“先验知识”字典。
    """
    print(f"正在读取文件: {filepath} ...")
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 未找到。请确保文件路径正确。")
        return None

    print("数据准备与清洗...")
    def parse_vector(val):
        try:
            vector = ast.literal_eval(str(val))
            if isinstance(vector, list) and len(vector) == 3:
                return [float(v) for v in vector]
        except (ValueError, SyntaxError, TypeError):
            return None
    
    df['ipp_vector'] = df['ImagePositionPatient'].apply(parse_vector)
    df.dropna(subset=['ipp_vector'], inplace=True)
    df[['ipp_x', 'ipp_y', 'ipp_z']] = pd.DataFrame(df['ipp_vector'].tolist(), index=df.index)

    multi_part_mask = df['root'].str.contains(',', na=False)
    multi_part_types = df[multi_part_mask]['root'].unique()
    
    if len(multi_part_types) == 0:
        print("未在'root'列中找到任何多部位检查。")
        return {}

    print(f"发现 {len(multi_part_types)} 种多部位检查类型: {list(multi_part_types)}")
    
    anatomical_priors = {}

    for part_string in multi_part_types:
        print(f"\n正在处理类型: '{part_string}' ...")
        df_subset = df[df['root'] == part_string].copy()
        k = len(part_string.split(','))
        
        # 确定用于聚类的坐标轴
        if df_subset.empty or k < 2:
            print(f"  - 跳过 '{part_string}'，因为序列数量不足或不是多部位。")
            continue
            
        distinguishing_axis = 'X' if '膝' in part_string else 'Z'
        print(f"  - 聚类数量 (K): {k}, 区分轴: {distinguishing_axis}")

        clustering_data = df_subset[[f'ipp_{distinguishing_axis.lower()}']].values
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df_subset['cluster_id'] = kmeans.fit_predict(clustering_data)
        
        clusters_list = []
        for cluster_id in range(k):
            cluster_df = df_subset[df_subset['cluster_id'] == cluster_id]
            if cluster_df.empty:
                continue
            
            series_count = len(cluster_df)
            avg_pos_vector = np.mean(np.stack(cluster_df['ipp_vector']), axis=0)
            avg_pos_dict = {"X": round(avg_pos_vector[0], 2), "Y": round(avg_pos_vector[1], 2), "Z": round(avg_pos_vector[2], 2)}
            distinguishing_axis_value = round(cluster_df[f'ipp_{distinguishing_axis.lower()}'].mean(), 2)
            
            clusters_list.append({
                "cluster_id": cluster_id, "series_count": series_count, "avg_position": avg_pos_dict,
                "distinguishing_axis_value": distinguishing_axis_value, "anatomical_label": "PLEASE_FILL_IN"
            })
        
        description = (f"通常包含{k}个空间簇，可通过{distinguishing_axis}轴位置进行区分。"
                       f"{'X轴从右向左为正方向。' if distinguishing_axis == 'X' else 'Z值越大，位置越靠近头顶。'}")
        
        clusters_list.sort(key=lambda x: x['distinguishing_axis_value'], reverse=True)

        anatomical_priors[part_string] = {
            "description": description, "distinguishing_axis": distinguishing_axis, "clusters": clusters_list
        }
    return anatomical_priors

def save_priors_to_file(priors_dict, output_filename):
    """
    将生成的先验知识字典格式化并保存到Python文件中。
    """
    if not priors_dict:
        print("没有可保存的先验知识数据。")
        return
        
    print(f"\n正在将字典写入文件: {output_filename} ...")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("# ==============================================================================\n")
        f.write("# MRI多部位扫描 先验知识配置字典\n")
        f.write("# ==============================================================================\n")
        f.write("# 说明:\n")
        f.write("# 此文件由generate_priors.py自动生成。\n")
        f.write("# 请根据每个'cluster'的'avg_position'和'distinguishing_axis_value'，\n")
        f.write("# 手工修改 'anatomical_label' 字段，填入正确的标准化部位名称。\n")
        f.write("# ==============================================================================\n\n")
        
        f.write("ANATOMICAL_PRIORS = {\n")
        for i, (key, value) in enumerate(priors_dict.items()):
            f.write(f'    "{key}": {{\n')
            f.write(f'        "description": "{value["description"]}",\n')
            f.write(f'        "distinguishing_axis": "{value["distinguishing_axis"]}",\n')
            f.write(f'        "clusters": [\n')
            for j, cluster in enumerate(value["clusters"]):
                f.write(f'            {{\n')
                f.write(f'                "cluster_id": {cluster["cluster_id"]},\n')
                f.write(f'                "series_count": {cluster["series_count"]},\n')
                f.write(f'                "avg_position": {cluster["avg_position"]},\n')
                f.write(f'                "distinguishing_axis_value": {cluster["distinguishing_axis_value"]},\n')
                f.write(f'                "anatomical_label": "{cluster["anatomical_label"]}"\n')
                f.write(f'            }}{"," if j < len(value["clusters"]) - 1 else ""}\n')
            f.write(f'        ]\n')
            f.write(f'    }}{"," if i < len(priors_dict.keys()) - 1 else ""}\n')
        f.write("}\n")
    print(f"成功！配置文件已保存至 '{output_filename}'。")

if __name__ == '__main__':
    # 定义输入和输出文件路径
    input_filepath = 'combined_data.xlsx'
    output_filepath = 'anatomical_priors_config.py'
    
    # 执行主函数
    final_priors_dict = generate_anatomical_priors_dict(input_filepath)
    
    # 将结果保存到文件
    save_priors_to_file(final_priors_dict, output_filepath)