import json
from tqdm import tqdm

path = './difficulty/qwen/ccl_train_base_qwen.json'
easy_path = path.replace('.json','_easy.json')
hard_path = path.replace('.json','_hard.json')

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
# res = [0,0,0,0,0,0,0,0,0,0,0]
easy = []
hard = []
instruction = '比喻是一种修辞手法，通过暗示或类比来传达某种含义，在比喻中，一个概念或对象通常被用来表示另一个概念或对象。请根据以上概念判断以下句子中是否包含比喻，只需回答是或否：'

for item in tqdm(data):
    cnt = 0
    inputs = item['input']
    responses = item['responses']
    label = item['label']
    # for i in range(len(responses)):
    for response in responses:
        if response[0] == label:
            cnt+=1
    # res[cnt] += 1
    if cnt >= 5:
        easy.append ({'instruction':instruction,'input': inputs, 'output': label})
    else:
        hard.append ({'instruction':instruction,'input': inputs, 'output': label})
# print(res)
with open(easy_path, 'w', encoding='utf-8') as json_file:
    json.dump(easy, json_file, indent=2, ensure_ascii=False)
with open(hard_path, 'w', encoding='utf-8') as json_file:
    json.dump(hard, json_file, indent=2, ensure_ascii=False)
