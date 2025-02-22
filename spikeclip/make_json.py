import os
import json

def generate_json_from_folder(base_path, output_json_path):
    """
    遍历给定路径下的所有子文件夹，为每个文件夹生成路径和对应的标签（文件夹名称）。
    
    :param base_path: 数据集的根目录
    :param output_json_path: 生成的 JSON 文件路径
    """
    folder_label_mapping = {}

    # 遍历给定路径下的所有文件夹
    for root, dirs, files in os.walk(base_path):
        # 排除不包含子文件夹的路径
        if not dirs:
            # 生成文件夹路径和标签
            label = os.path.basename(os.path.dirname(root))  # 获取父文件夹作为标签
            folder_label_mapping[root] = label

    # 将字典保存为 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(folder_label_mapping, json_file, ensure_ascii=False, indent=4)

    print(f"JSON 文件已保存到 {output_json_path}")

if __name__ == "__main__":
    # 给定数据集路径
    base_path = "./video_s"
    # 输出 JSON 文件路径
    output_json_path = "./UCF101_3_10_.json"
    
    # 生成 JSON 文件
    generate_json_from_folder(base_path, output_json_path)

