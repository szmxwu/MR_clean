# ==============================================================================
# MRI多部位扫描 先验知识配置字典
# ==============================================================================
# 说明:
# 此文件由generate_priors.py自动生成。
# 请根据每个'cluster'的'avg_position'和'distinguishing_axis_value'，
# 手工修改 'anatomical_label' 字段，填入正确的标准化部位名称。
# ==============================================================================

ANATOMICAL_PRIORS = {
    "腹部,盆部": {
        "description": "通常包含2个空间簇，可通过Z轴位置进行区分。Z值越大，位置越靠近头顶。",
        "distinguishing_axis": "Z",
        "clusters": [
            {
                "cluster_id": 0,
                "series_count": 3899,
                "avg_position": {'X': -186.74, 'Y': -83.73, 'Z': 28.33},
                "distinguishing_axis_value": 28.33,
                "anatomical_label": "腹部"
            },
            {
                "cluster_id": 1,
                "series_count": 1878,
                "avg_position": {'X': -144.43, 'Y': -107.49, 'Z': -227.38},
                "distinguishing_axis_value": -227.38,
                "anatomical_label": "盆部"
            }
        ]
    },
    "脑神经,颅脑": {
        "description": "通常包含2个空间簇，可通过Z轴位置进行区分。Z值越大，位置越靠近头顶。",
        "distinguishing_axis": "Z",
        "clusters": [
            {
                "cluster_id": 0,
                "series_count": 11,
                "avg_position": {'X': -83.01, 'Y': -66.16, 'Z': 78.43},
                "distinguishing_axis_value": 78.43,
                "anatomical_label": "颅脑"
            },
            {
                "cluster_id": 1,
                "series_count": 5,
                "avg_position": {'X': -94.96, 'Y': -61.72, 'Z': -27.54},
                "distinguishing_axis_value": -27.54,
                "anatomical_label": "脑神经"
            }
        ]
    }
}
