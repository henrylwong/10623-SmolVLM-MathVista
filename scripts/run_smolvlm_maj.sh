cd ../evaluation

##### smolvlm #####
# generate solution
# python generate_response.py \
# --model smolvlm \
# --output_dir ../results/smolvlm/maj \
# -output_file output_smolvlm.json \
# --run_maj 5

# extract answer
python extract_smolvlm_answer.py \
--output_dir ../results/smolvlm/maj \
--output_file output_smolvlm.json 

# calculate score
python calculate_score.py \
--output_dir ../results/smolvlm/maj \
--output_file output_smolvlm.json \
--score_file scores_smolvlm.json
