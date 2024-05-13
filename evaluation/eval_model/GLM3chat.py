from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("/data/huboxiang/myllm/THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/data/huboxiang/myllm/THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
model = model.eval()

with open('cme_metaphor_test_base.json', 'r', encoding='utf-8') as f:
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
    except:
        print("ERROR: Unexpected response format or API error:", context)
        continue

with open('cme_metaphor_test_base_chatglm3.json', 'w', encoding='utf-8') as json_file:
    json.dump(out, json_file, indent=2, ensure_ascii=False)