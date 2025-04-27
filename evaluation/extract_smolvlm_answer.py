import argparse
import logging
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.logging import RichHandler
from tqdm import tqdm
import pdb

from evaluation.prompts.ext_ans import demo_prompt
from utilities import read_json, save_json

CACHE_DIR = "hf_cache"

def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True

def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt

def extract_answer(model, response, problem, tokenizer, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""

    if question_type == 'multi_choice' and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            
            return str(extraction)
        except Exception as e:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            
            return extraction
        except Exception as e:
            pass

    # Quick extraction: regex string is extremely strict
    if quick_extract:
        logging.info("Quickly extracting answer...")
        try:
            result = re.search(r'The answer is "(.*)"\.?', response)
            if result:
                extraction = result.group(1)
                return extraction
        except Exception as e:
            pass

    # General extraction
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": full_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        extraction = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        extraction = extraction.split('</think>') # @hlwong: verify
        return extraction[-1]
    except Exception as e:
        logging.info(f"Error in extracting answer for problem: {pid} with response: {response}")
        logging.info(e)

    return ""


def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--results_file_path', type=str, default='output_smolvlm.json')
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    parser.add_argument('--max_num_problems', type=int, default=-1, help='The max number of problems to run')
    parser.add_argument('--quick_extract', action='store_true', help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    # output
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')
    parser.add_argument('--output_dir', type=str, default='_results/eval/mathvista/llava/debug')
    parser.add_argument('--output_file', type=str)

    args = parser.parse_args()
    return args

def main():
    logging.info("MathVista: Extract Answers - Start")
    args = parse_args()

    # args
    label = args.response_label

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=CACHE_DIR
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, CACHE_DIR=CACHE_DIR)

    results_file_path = os.path.join(args.output_dir, args.output_file)
    logging.info(f"Reading {results_file_path}...")
    results = read_json(results_file_path)

    full_pids = list(results.keys())
    pdb.set_trace()
    skip_pids = []
    for pid, problem in results.items():
        extraction = problem.get('extraction')
        if extraction is not None and verify_extraction(extraction):
            skip_pids.append(problem['pid'])

    if args.rerun:
        test_pids = full_pids
    else:
        if len(skip_pids) > 0:
            logging.info(
                f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems..."
            )
        test_pids = [pid for pid in full_pids if pid not in skip_pids]

    if args.max_num_problems > 0:
        test_pids = test_pids[: min(args.max_num_problems, len(test_pids))]
        logging.info(f'Limiting number of problems to {args.max_num_problems}.')

    logging.info(f"Number of test problems to run: {len(test_pids)}")
    print("test len", len(test_pids))
    for i, pid in enumerate(tqdm(test_pids)):
        problem = results[pid]

        assert label in problem
        response = problem[label]
        extraction = extract_answer(model, response, problem, tokenizer, args.quick_extract)
        results[pid]['extraction'] = extraction

        if (i % args.save_every == 0 and i > 0) or i == len(test_pids) - 1:
            save_json(results, results_file_path)
            logging.info(f"Saved results to {results_file_path}")

    logging.info("MathVista: Extract Answers - Finish")


if __name__ == '__main__':
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                omit_repeated_times=False,
            )
        ],
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()