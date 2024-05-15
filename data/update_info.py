import json
import os
import hashlib

def calculate_sha1(file_path):
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as file:
        while True:
            data = file.read(65536)  # 64KB buffer
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()

info_path = 'dataset_info.json'

current_folder = os.getcwd()
 
# 列出当前文件夹下所有文件和文件夹的名称
file_names = os.listdir(current_folder)
 
info = {}

# 打印所有文件名
for file_name in file_names:
    if file_name.endswith('.json'):
        if file_name == info_path:
            continue
        else:
            info[file_name[0:-5]] = {"file_name":file_name,"file_sha1": calculate_sha1(current_folder+'/'+file_name) }
print(info)
with open(info_path, 'w', encoding='utf-8') as json_file:
    json.dump(info, json_file, indent=2, ensure_ascii=False)