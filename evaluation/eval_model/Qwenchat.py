from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
device = "cuda" # the device to load the model onto

model_path = '/data/huboxiang/metaphor/LLaMA-Factory/saves/Qwen1.5-7B-Chat/full/sft-ccl_train_base'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
str1 = '比喻是一种修辞手法，通过暗示或类比来传达某种含义，在比喻中，一个概念或对象通常被用来表示另一个概念或对象。请根据以上概念判断以下句子中是否包含比喻，只回答是或否：'
out = []
with open('ccl_test_base.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in tqdm(data):
    context = item['input']
    label = item['output']
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": str1+context}
    ]
    try :
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        out.append({'input':item['input'],'label': label, 'response': response})
        print(f'label:{label},response:{response}')
        print(response)
    except:
        print("ERROR: Unexpected response format or API error:", context)
        continue

with open('ccl_test_sft-ccl-train_Qwen-7b-chat.json', 'w', encoding='utf-8') as json_file:
    json.dump(out, json_file, indent=2, ensure_ascii=False)