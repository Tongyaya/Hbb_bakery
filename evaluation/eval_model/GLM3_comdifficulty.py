from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
device = "cuda" # the device to load the model onto

model_path = "/data/huboxiang/myllm/chatglm3-6b"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = model.eval()

test_data_path = './test_data/ccl_train_base.json'
model_output_path = test_data_path.replace('./test_data/','./difficulty/glm3/').replace('.json','_glm3.json')
with open(test_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

str1 = '比喻是一种修辞手法，通过暗示或类比来传达某种含义，在比喻中，一个概念或对象通常被用来表示另一个概念或对象。请根据以上概念判断以下句子中是否包含比喻，只需回答是或否：'
out = []

for item in tqdm(data):
    context = item['input']
    label = item['output']
    responses = []
# str2 = '音 越拉越长，越细，越尖锐 象山丘的轮廓终于平伏 你身体的线条也不再弯曲 象一条抽象的直线越出了这张纸'
    for i in range(10):
        cnt = 0
        while 1 :
            try :
                response, history = model.chat(tokenizer, str1+context, history=[])
                if response[0]=='是' or response[0]=='否':
                    break
                cnt += 1
                if cnt >= 3:
                    if label == '是':
                        response = '否'
                    elif label == '否':
                        response = '是'
                    break
                # out.append({'input':item['input'],'label': label, 'response': response})
                # print(f'label:{label},response:{response}')
            except Exception as e:
                print("ERROR:", e)
                print(context)
                continue
        responses.append(response)
    out.append({'input':item['input'],'label': label, 'responses': responses}) 
    

with open(model_output_path, 'w', encoding='utf-8') as json_file:
    json.dump(out, json_file, indent=2, ensure_ascii=False)

print(f'Saved in {model_output_path}')