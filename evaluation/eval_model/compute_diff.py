import json
from tqdm import tqdm

path = './difficulty/vicuna/VUA18_train_vicuna-7b-v1.5-0.json'
easy_path = path.replace('.json','_easy.json')
hard_path = path.replace('.json','_hard.json')
medi_path = path.replace('.json','_medi.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
res = [0,0,0,0,0,0,0,0,0,0,0]
easy = []
hard = []
medi = []
# instruction = '''辨别隐喻通常遵循以下步骤：
# （a） 分析并确定话语语境中每个词汇单元的含义。
# （b） 确定每个词汇单元在其他语境中是否比在给定的话语语境中具有更基本的含义。
# （c） 如果词汇单元在其他上下文中比在当前上下文中具有更基本的含义，则确定上下文含义是否与基本含义形成对比，并与之相比能够理解上下文含义。
# （d） 如果是，那么将这个词汇单位标记为隐喻。
# 使用以上步骤来确定以下句子是否包含隐喻，只回答是或否：'''
instruction = '''Discriminating metaphors usually follows the following steps:
(a) Analyze and determine the meaning of each lexical unit in the discourse context.
(b) Determine whether each lexical unit has a more basic meaning in other contexts than in the given discourse context.
(c) If the lexical unit has a more basic meaning in other contexts than in the current context, determine whether the contextual meaning contrasts with the basic meaning and can understand the contextual meaning in comparison with it.
(d) If so, then mark this lexical unit as metaphorical.
Use the above steps to determine whether the following sentences contain metaphors.
Answering starts with ''yes'' or ''no'' :'''

for item in tqdm(data):
    cnt = 0
    inputs = item['input']
    responses = item['responses']
    label = item['label']
    # for i in range(len(responses)):
    for response in responses:
        if response == label:
            cnt+=1
    res[cnt] += 1
    if cnt == 10:
        easy.append ({'instruction':instruction,'input': inputs, 'output': label})
    elif cnt == 0:
        hard.append ({'instruction':instruction,'input': inputs, 'output': label})
    else:
        medi.append ({'instruction':instruction,'input': inputs, 'output': label})
print(res)
with open(easy_path, 'w', encoding='utf-8') as json_file:
    json.dump(easy, json_file, indent=2, ensure_ascii=False)
with open(hard_path, 'w', encoding='utf-8') as json_file:
    json.dump(hard, json_file, indent=2, ensure_ascii=False)
with open(medi_path, 'w', encoding='utf-8') as json_file:
    json.dump(medi, json_file, indent=2, ensure_ascii=False)
