"""
Standard and CoT prompts are used during generation.
 - Standard: generate final output in one shot
 - CoT: limit generation to step-by-step
Value prompts are used during evaluation / selection.

For few-shot, add samples from MathVista set or similar. Add restraint that it is only allowed to generate one step at a time.
Add max step of N = 10. If final iteration, maybe just ask model to generate answer.
"""

standard_prompt = '''You are given a math problem. Solve it step by step, writing each step on a new line. When you reach the final answer, write it on a new line starting with "Answer:".

{input}

All steps:
'''

# cot_first_prompt = '''Solve the math problem step by step. For each step, you may write a very short phrase (if needed for clarity), then the calculation. Do not number the steps. Do not write full sentences or explanations. Only use a short phrase if it helps keep track of the calculation. If you know the answer, write it as "Answer: ...".

# Problem: A class has 28 students. 60% of them passed the math test. Later, 5 more students joined the class, and all of them passed the test. How many students in total have passed the math test now?
# Steps:
# Passed: 28 * 0.6 = 16.8
# Round: round(16.8) = 17
# New total: 28 + 5 = 33
# Total passed: 17 + 5 = 22
# Answer: 22

# Problem: A car travels 180 kilometers in 3 hours, then 120 kilometers in 2 hours. What is the car's average speed in kilometers per hour for the whole trip?
# Steps:
# Total distance: 180 + 120 = 300
# Total time: 3 + 2 = 5
# Average speed: 300 / 5 = 60
# Answer: 60

# Problem: A recipe requires 2.5 cups of flour for one batch of cookies. If you want to make 3 batches, but you only have a 1/2 cup measuring cup, how many times do you need to fill the 1/2 cup to get enough flour?
# Steps:
# Total flour: 2.5 * 3 = 7.5
# Number of fills: 7.5 / 0.5 = 15
# Answer: 15

# Problem:
# {input}

# First step:
# '''

# cot_prompt = '''Solve the math problem step by step. For each step, you may write a very short phrase (if needed for clarity), then the calculation. Do not number the steps. Do not write full sentences or explanations. Only use a short phrase if it helps keep track of the calculation. If you know the answer, write it as "Answer: ...".

# Problem: A class has 28 students. 60% of them passed the math test. Later, 5 more students joined the class, and all of them passed the test. How many students in total have passed the math test now?
# Steps:
# Passed: 28 * 0.6 = 16.8
# Round: round(16.8) = 17
# New total: 28 + 5 = 33
# Total passed: 17 + 5 = 22
# Answer: 22

# Problem: A car travels 180 kilometers in 3 hours, then 120 kilometers in 2 hours. What is the car's average speed in kilometers per hour for the whole trip?
# Steps:
# Total distance: 180 + 120 = 300
# Total time: 3 + 2 = 5
# Average speed: 300 / 5 = 60
# Answer: 60

# Problem: A recipe requires 2.5 cups of flour for one batch of cookies. If you want to make 3 batches, but you only have a 1/2 cup measuring cup, how many times do you need to fill the 1/2 cup to get enough flour?
# Steps:
# Total flour: 2.5 * 3 = 7.5
# Number of fills: 7.5 / 0.5 = 15
# Answer: 15

# Problem:
# {input}

# Current steps:
# {steps}

# Next step or Answer:
# '''

# cot_last_prompt = '''Solve the math problem step by step. Given the problem and your reasoning steps so far, provide the final answer. For each step, you may write a very short phrase (if needed for clarity), then the calculation. Do not number the steps. Do not write full sentences or explanations. Only use a short phrase if it helps keep track of the calculation. Write the answer as "Answer: ...".

# Problem: A class has 28 students. 60% of them passed the math test. Later, 5 more students joined the class, and all of them passed the test. How many students in total have passed the math test now?
# Steps:
# Passed: 28 * 0.6 = 16.8
# Round: round(16.8) = 17
# New total: 28 + 5 = 33
# Total passed: 17 + 5 = 22
# Answer: 22

# Problem: A car travels 180 kilometers in 3 hours, then 120 kilometers in 2 hours. What is the car's average speed in kilometers per hour for the whole trip?
# Steps:
# Total distance: 180 + 120 = 300
# Total time: 3 + 2 = 5
# Average speed: 300 / 5 = 60
# Answer: 60

# Problem: A recipe requires 2.5 cups of flour for one batch of cookies. If you want to make 3 batches, but you only have a 1/2 cup measuring cup, how many times do you need to fill the 1/2 cup to get enough flour?
# Steps:
# Total flour: 2.5 * 3 = 7.5
# Number of fills: 7.5 / 0.5 = 15
# Answer: 15

# Problem:
# {input}

# Current steps:
# {steps}

# Answer:
# '''

# cot_first_prompt = '''Solve the math problem step by step. Write only one major calculation per line. Do not combine steps. If you know the answer, write it as "Answer: ...".

# Problem: A class has 28 students. 60% of them passed the math test. Later, 5 more students joined the class, and all of them passed the test. How many students in total have passed the math test now?
# Steps:
# 28 * 0.6 = 16.8
# round(16.8) = 17
# 28 + 5 = 33
# 17 + 5 = 22
# Answer: 22

# Problem: A car travels 180 kilometers in 3 hours, then 120 kilometers in 2 hours. What is the car's average speed in kilometers per hour for the whole trip?
# Steps:
# 180 + 120 = 300
# 3 + 2 = 5
# 300 / 5 = 60
# Answer: 60

# Problem: A recipe requires 2.5 cups of flour for one batch of cookies. If you want to make 3 batches, but you only have a 1/2 cup measuring cup, how many times do you need to fill the 1/2 cup to get enough flour?
# Steps:
# 2.5 * 3 = 7.5
# 7.5 / 0.5 = 15
# Answer: 15

# Problem: {input}

# First step:
# '''

# cot_prompt = '''Solve the math problem step by step. Write only one major calculation per line. Do not combine steps. If you know the answer, write it as "Answer: ...".

# Problem: A class has 28 students. 60% of them passed the math test. Later, 5 more students joined the class, and all of them passed the test. How many students in total have passed the math test now?
# Steps:
# 28 * 0.6 = 16.8
# round(16.8) = 17
# 28 + 5 = 33
# 17 + 5 = 22
# Answer: 22

# Problem: A car travels 180 kilometers in 3 hours, then 120 kilometers in 2 hours. What is the car's average speed in kilometers per hour for the whole trip?
# Steps:
# 180 + 120 = 300
# 3 + 2 = 5
# 300 / 5 = 60
# Answer: 60

# Problem: A recipe requires 2.5 cups of flour for one batch of cookies. If you want to make 3 batches, but you only have a 1/2 cup measuring cup, how many times do you need to fill the 1/2 cup to get enough flour?
# Steps:
# 2.5 * 3 = 7.5
# 7.5 / 0.5 = 15
# Answer: 15

# Problem: {input}

# Current steps:
# {steps}

# Next step or Answer:
# '''

# cot_last_prompt = '''Solve the math problem step by step. Given the problem and your reasoning steps so far, provide the final answer. Write only one major calculation per line. Do not combine steps. Write the answer as "Answer: ...".

# Problem: A class has 28 students. 60% of them passed the math test. Later, 5 more students joined the class, and all of them passed the test. How many students in total have passed the math test now?
# Steps:
# 28 * 0.6 = 16.8
# round(16.8) = 17
# 28 + 5 = 33
# 17 + 5 = 22
# Answer: 22

# Problem: A car travels 180 kilometers in 3 hours, then 120 kilometers in 2 hours. What is the car's average speed in kilometers per hour for the whole trip?
# Steps:
# 180 + 120 = 300
# 3 + 2 = 5
# 300 / 5 = 60
# Answer: 60

# Problem: A recipe requires 2.5 cups of flour for one batch of cookies. If you want to make 3 batches, but you only have a 1/2 cup measuring cup, how many times do you need to fill the 1/2 cup to get enough flour?
# Steps:
# 2.5 * 3 = 7.5
# 7.5 / 0.5 = 15
# Answer: 15

# Problem: {input}

# Current steps:
# {steps}

# Answer:
# '''

cot_first_prompt = '''You are given a math problem. Begin solving it step by step. Write only the first reasoning step on a new line. Do not provide the final answer yet.

Problem:
{input}

First step:
'''

cot_prompt = '''You are solving a math problem step by step. Given the problem and your current reasoning steps so far, write the next logical step on a new line. 
If you have enough information to solve the problem, write the final answer on a new line starting with "Answer:". 
Think carefully and make sure each step follows logically from the previous ones.

Problem:
{input}

Current steps:
{steps}

Next step or Answer:
'''

cot_last_prompt = '''You are solving a math problem step by step. Given the problem and your reasoning steps so far, provide the final answer. Write the answer on a new line starting with "Answer:".

Problem:
{input}

Solution steps:
{steps}

Answer:
'''

propose_prompt = '''Given the following math problem and the current solution steps, propose possible next steps to the solution. Only provide possible next steps, each on a new line, and do not solve the entire problem. When you reach the final answer, write it on a new line starting with "Answer:".

Problem:
{input}

Current steps:
{steps}

Possible next steps:
'''

value_prompt = '''Given the following math problem and the current solution steps, how likely is it that the steps will lead to the correct answer?

Problem: {input}

Current steps:
{steps}

Respond with one of: impossible, unlikely, possible, likely, sure.
'''

value_last_step_prompt = '''Given the following math problem, solution steps, and provided answer, how likely is it that the answer is correct?

Problem: {input}

Current steps:
{steps}

Answer:
{answer}

Respond with one of: impossible, unlikely, possible, likely, sure.
'''