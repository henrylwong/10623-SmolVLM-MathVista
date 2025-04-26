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

def get_samples(model_infer, task, user_prompt, decoded_image, response, prompt_sample, num_samples, step=None, max_depth=None):
    if step is None or max_depth is None:
        raise ValueError("get_samples(): value or max_depth are not defined.")

    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(user_prompt, response)
    elif prompt_sample == 'cot':
        if step == 0:
            prompt = task.cot_first_prompt_wrap(user_prompt, response)
        elif step == max_depth - 1:
            prompt = task.cot_last_prompt_wrap(user_prompt, response)
        else:
            prompt = task.cot_prompt_wrap(user_prompt, response)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = model_infer.get_multiple_responses(prompt, decoded_image, num_samples)
    return [response + '\n' + sample for sample in samples]

def solve(model_infer, task, user_prompt, decoded_image, debug=True):
    ys = list([""]) # maintain current ouput candidates
    results = list()
    infos = list()
    for step in range(task.max_depth):
        # Separate finished and unfinished paths
        unfinished_ys = list()
        for y in ys:
            if "answer" in y.lower():
                results.append(y)
            else:
                unfinished_ys.append(y)
        if not unfinished_ys:
            break

        # Generation (sample-based)
        new_ys = list()
        for y in ys:
            new_ys.extend(get_samples(
                model_infer, task, user_prompt, decoded_image, y, "cot", NUM_GENERATE_SAMPLES,
                step = step, max_depth=task.max_depth
            ))
        ids = list(range(len(new_ys)))
        # pdb.set_trace()

        # Evaluation (values-based)
        values = get_values(model_infer, task, user_prompt, decoded_image, new_ys)

        # Begin Value Filtering
        # If any value is "sure" or "likely", keep only those
        # Otherwise, if any value is non-zero, filter out all zero values
        thresholds = [1, 0.001]  # likely, impossible
        for thresh in thresholds:
            idxs = [i for i, v in enumerate(values) if v >= thresh]
            if idxs:
                new_ys = [new_ys[i] for i in idxs]
                values = [values[i] for i in idxs]
                ids = list(range(len(new_ys)))
                break

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
        print(f'All ys: {ys}\n\n') 

    def select_best(candidates):
        if not candidates:
            return ""
        vals = get_values(model_infer, task, user_prompt, decoded_image, candidates)
        best_idx = np.argmax(vals)
        if debug:
            print(f'Best answer: {candidates[best_idx]} (value: {vals[best_idx]})')
        return candidates[best_idx]

    # Prefer finished results, otherwise best unfinished
    best_answer = select_best(results) if results else select_best(ys)
    pdb.set_trace()
    return best_answer, {'steps': infos}
