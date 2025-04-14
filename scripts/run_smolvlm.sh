cd ../evaluation

##### smolvlm #####
# generate solution
python generate_response.py \
--model smolvlm \
--output_dir ../results/smolvlm \
--output_file out/output_smolvlm.json

exit 0

# extract answer
python extract_answer.py \
--output_dir ../results/smolvlm \
--output_file output_smolvlm.json 

# calculate score
python calculate_score.py \
--output_dir ../results/smolvlm \
--output_file output_smolvlm.json \
--score_file scores_smolvlm.json
