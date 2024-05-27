# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
device = "cuda" # the device to load the model onto

model_path = '/data/huboxiang/myllm/llama2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()

instruction = '''Discriminating metaphors usually follows the following steps:
(a) Analyze and determine the meaning of each lexical unit in the discourse context.
(b) Determine whether each lexical unit has a more basic meaning in other contexts than in the given discourse context.
(c) If the lexical unit has a more basic meaning in other contexts than in the current context, determine whether the contextual meaning contrasts with the basic meaning and can understand the contextual meaning in comparison with it.
(d) If so, then mark this lexical unit as metaphorical.
Use the above steps to determine whether the following sentences contain metaphors.
Answering starts with ''yes'' or ''no'' :'''


out = []
test_data_path = './test_data/MOH-X_train.json'
model_output_path = test_data_path.replace('.json','_llama2-7b-chat-hf.json').replace('./test_data/','./difficulty/llama2-2/')
with open(test_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
responses = []
for item in tqdm(data):
    context = item['input']
    label = item['output']
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction+context}
        ]
    cnt = 0
    # for i in range(10):
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
    while 1 :
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if response[:3].lower() == "yes" or response[:2].lower() == "no" or response[-2:].lower() == "no" or response[-3:].lower() == "yes":
                break
        cnt += 1
        if cnt >= 2:
            break
    print(f'response is {response}')
    out.append({'input':item['input'],'label': label, 'response': response})

    with open(model_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(out, json_file, indent=2, ensure_ascii=False)

print(f'model path is {model_path}')
print(f'Saved in {model_output_path}')