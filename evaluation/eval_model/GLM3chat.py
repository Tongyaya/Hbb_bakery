from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = "5"
device = "cuda" # the device to load the model onto

model_path = "/data/huboxiang/metaphor/Hbb_Factory/saves/chatglm3-6b/full/sft-meta_1500-CL-train"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = model.eval()

test_data_path = './test_data/ccl_test_base.json'
model_output_path = test_data_path.replace('.json','_sft-meta_1500-CL-train-chatglm3-6b.json').replace('test_data','model_output')
with open(test_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

str1 = '''辨别隐喻通常遵循以下步骤：
（a） 分析并确定话语语境中每个词汇单元的含义。
（b） 确定每个词汇单元在其他语境中是否比在给定的话语语境中具有更基本的含义。
（c） 如果词汇单元在其他上下文中比在当前上下文中具有更基本的含义，则确定上下文含义是否与基本含义形成对比，并与之相比能够理解上下文含义。
（d） 如果是，那么将这个词汇单位标记为隐喻。
使用以上步骤来确定以下句子是否包含隐喻，只回答是或否：'''

out = []
for item in tqdm(data):
    context = item['input']
    label = item['output']

# str2 = '音 越拉越长，越细，越尖锐 象山丘的轮廓终于平伏 你身体的线条也不再弯曲 象一条抽象的直线越出了这张纸'
    str2 = '金色的秋天'
    while 1 :
        try :
            response, history = model.chat(tokenizer, str1+str2, history=[])
            # if response[0]=='是' or response[0]=='否':
            #     break
            
            print(f'response:{response}')
        except Exception as e:
            print("ERROR:", e)
            continue
    print(f'label: {label}, response: {response}')
    out.append({'input':item['input'],'label': label, 'response': response})

with open(model_output_path, 'w', encoding='utf-8') as json_file:
    json.dump(out, json_file, indent=2, ensure_ascii=False)

print(f'Saved in {model_output_path}')