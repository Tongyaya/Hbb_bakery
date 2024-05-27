# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import transformers
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
device = "cuda" # the device to load the model onto

model_path = '/data/huboxiang/myllm/llama3-8b-instruct'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

instruction = '''Discriminating metaphors usually follows the following steps:
(a) Analyze and determine the meaning of each lexical unit in the discourse context.
(b) Determine whether each lexical unit has a more basic meaning in other contexts than in the given discourse context.
(c) If the lexical unit has a more basic meaning in other contexts than in the current context, determine whether the contextual meaning contrasts with the basic meaning and can understand the contextual meaning in comparison with it.
(d) If so, then mark this lexical unit as metaphorical.
Use the above steps to determine whether the following sentences contain metaphors.
Answer starts with "yes" or "no" :'''


out = []
test_data_path = './test_data/MOH-X_train.json'
model_output_path = test_data_path.replace('.json','_llama3-8b-instruct.json').replace('./test_data/','./difficulty/llama3-8/')
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
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    # for i in range(10):
    while 1 :
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        if response[:3].lower() == "yes" or response[:2].lower() == "no" or response[-2:].lower() == "no" or response[-3:].lower() == "yes":
                break
    print(response)
    out.append({'input':item['input'],'label': label, 'response': response})

    with open(model_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(out, json_file, indent=2, ensure_ascii=False)

print(f'model path is {model_path}')
print(f'Saved in {model_output_path}')