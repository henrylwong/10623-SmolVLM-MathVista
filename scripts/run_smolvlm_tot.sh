cd ../evaluation

##### smolvlm #####
# generate solution
python generate_response.py \
--model smolvlm \
--output_dir ../results/smolvlm/tot \
--output_file output_smolvlm.json \
--run_tot

exit 0

# extract answer
python extract_answer.py \
--output_dir ../results/smolvlm/tot \
--output_file output_smolvlm.json 

# calculate score
python calculate_score.py \
--output_dir ../results/smolvlm/tot \
--output_file output_smolvlm.json \
--score_file scores_smolvlm.json
