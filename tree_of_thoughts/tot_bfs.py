import numpy as np
import itertools
import pdb

METHOD_SELECT = 'greedy'
NUM_GENERATE_SAMPLES = 2
NUM_SELECT_SAMPLE = 2

def get_value(model_infer, task, user_prompt, decoded_image, y):
    value_prompt = task.value_prompt_wrap(user_prompt, y)
    value_output = model_infer.get_response(value_prompt, decoded_image)
    value = task.value_outputs_unwrap(y, value_output)
    return value

def get_values(model_infer, task, user_prompt, decoded_image, ys):
    values = []
    local_value_cache = {}
    # Evaluate each partial output
    for y in ys:
        value = 0 # avoid duplicate candidates
        if y not in local_value_cache:
            value = get_value(model_infer, task, user_prompt, decoded_image, y)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_samples(model_infer, task, user_prompt, decoded_image, response, prompt_sample, num_samples):
    pdb.set_trace()
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(user_prompt, response)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(user_prompt, response)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = model_infer.get_multiple_responses(prompt, decoded_image, num_samples)
    return [response + '\n' + sample for sample in samples]

def solve(model_infer, task, user_prompt, decoded_image, debug=True):
    ys = list([""]) # maintain current ouput candidates
    infos = list()
    for step in range(task.max_depth):
        # Generation (sample-based)
        new_ys = list()
        for y in ys:
            new_ys.extend(get_samples(model_infer, task, user_prompt, decoded_image, y, "cot", NUM_GENERATE_SAMPLES))
        ids = list(range(len(new_ys)))
        pdb.set_trace()

        # Evaluation (values-based)
        values = get_values(model_infer, task, user_prompt, decoded_image, new_ys)

        # Selection
        if METHOD_SELECT == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=NUM_SELECT_SAMPLE, p=ps).tolist()
        elif METHOD_SELECT == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:NUM_SELECT_SAMPLE]
        else:
            raise ValueError("Invalid selection method chosen (method_select).")
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # Logging
        if debug: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': user_prompt, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if debug: 
        print(ys)
    return ys, {'steps': infos}
