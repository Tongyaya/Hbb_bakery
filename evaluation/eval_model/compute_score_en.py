import json

# import jsonlines

def calculate_f1_score(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score,precision,recall

def scorecompute(file_path):
    prefix_path = './model_output/'
    
    debug_path = './debug/'+file_path.replace('.json','_debug.json')
    result_path = './result/'+file_path.replace('.json','_res.json')
    badcase_path = './badcase/'+file_path.replace('.json','_badcase.json')

    file_path = prefix_path+file_path
    # data = []
    tp = 0; fp = 0; fn = 0
    debug = []
    temp = []
    badcase = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        if item['label'] == item['response'] or item['label'] == item['response'][0]:
            tp += 1
        else:
            temp.append(item)
    for item in temp:
        if item['label'] == "yes" and (item['response'] == "no" or item['response'][0] == "no"):
            fn += 1
            badcase.append(item)
        elif item['label'] == "no" and (item['response'] == "yes" or item['response'][0] == "yes"):
            fp += 1
            badcase.append(item)
        else:
            debug.append(item)
    # tp += 4
    # fn += 14
    f1_score,precision,recall = calculate_f1_score(tp, fp, fn)

    res = [{'file_name':file_path,'f1_score':f1_score,'precision':precision,'recall':recall,'tp':tp,'fp':fp,"fn":fn,"total":len(data)}]

    print(f'tp:{tp},fp:{fp},fn:{fn},total:{len(data)},debug:{len(debug)}')

    print(f'f1_score,precision,recall:{f1_score,precision,recall}')

    def write2res(res,result_path):
        with open(result_path, 'w', encoding='utf-8') as json_file:
            json.dump(res, json_file, indent=2, ensure_ascii=False)
    write2res(debug,debug_path)
    write2res(badcase,badcase_path)
    write2res(res,result_path)

file_path = 'MOH-X_test_MOH-X_train_llama3-8b-instruct.json'
scorecompute(file_path)