from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
device = "cuda" # the device to load the model onto

model_path = "/data/huboxiang/metaphor/Hbb_Factory/saves/chatglm3-6b/full/sft-ccl_train_base"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = model.eval()

test_data_path = './test_data/ccl_test_base.json'
model_output_path = test_data_path.replace('.json','chatglm3-6b.json')
with open(test_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

str1 = '比喻是一种修辞手法，通过暗示或类比来传达某种含义，在比喻中，一个概念或对象通常被用来表示另一个概念或对象。请根据以上概念判断以下句子中是否包含比喻，只需回答是或否：'
out = []
for item in tqdm(data):
    context = item['input']
    label = item['output']

# str2 = '音 越拉越长，越细，越尖锐 象山丘的轮廓终于平伏 你身体的线条也不再弯曲 象一条抽象的直线越出了这张纸'
    try :
        response, history = model.chat(tokenizer, str1+context, history=[])
        out.append({'input':item['input'],'label': label, 'response': response})
        print(f'label:{label},response:{response}')
    except Exception as e:
        print("ERROR:", e)
        continue

with open(model_output_path, 'w', encoding='utf-8') as json_file:
    json.dump(out, json_file, indent=2, ensure_ascii=False)