from tqdm import tqdm
import os

def count_lines_in_python_files(directory):
    total_lines = 0
    # 使用 os.walk 遍历目录及其子目录
    for root, _, files in tqdm(os.walk(directory), desc=f"Traversing files in {directory}", total=len(os.listdir(directory))):
        for file in files:
            if file.endswith('.py'):  # 只统计 Python 文件
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_lines += sum(1 for _ in f)  # 统计文件中的行数
    return total_lines

# 使用示例
folder_path = './PyFDFD'  # 替换为你文件夹的路径
total_lines = count_lines_in_python_files(folder_path)
print(f"文件夹中的所有 Python 文件总行数: {total_lines}")

