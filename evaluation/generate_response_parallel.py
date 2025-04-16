import argparse
import os
import logging
from tqdm import tqdm

from datasets import load_dataset
from utilities import read_json, save_json
from utilities_aggregation import majority_vote
from models import smolvlm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='AI4Math/MathVista')
    parser.add_argument('--test_split_name', type=str, default='testmini')
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='../results/smolvlm_parallel')
    parser.add_argument('--output_file', type=str, default='output_majority.json')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_dataset(args.dataset_name, split=args.test_split_name)
    model = smolvlm.SmolVLMInfer()

    output_path = os.path.join(args.output_dir, args.output_file)
    results = {}

    for item in tqdm(data):
        pid = item['pid']
        decoded_img = item['decoded_image']
        query = item['query']

        candidates = model.get_multiple_responses(query, decoded_img, n=args.n_samples)
        final = majority_vote(candidates)

        results[pid] = {
            "query": query,
            "candidates": candidates,
            "majority_vote": final
        }

    save_json(results, output_path)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
