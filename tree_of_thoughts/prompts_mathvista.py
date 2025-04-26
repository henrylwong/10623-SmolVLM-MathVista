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

first_cot_prompt = '''You are given a math problem. Begin solving it step by step. Write only the first reasoning step on a new line. Do not provide the final answer yet.

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

last_cot_prompt = '''You are solving a math problem step by step. Given the problem and your reasoning steps so far, provide the final answer. Write the answer on a new line starting with "Answer:".

Problem:
{input}

Solution steps:
{steps}

Answer:
'''

propose_prompt = '''Given the following math problem and the current solution steps, propose possible next steps to the solution. Only provide possible next steps, each on a new line, and do not solve the entire problem. When you reach the final answer, write it on a new line starting with "Answer:".

{input}

Current steps:
{steps}

Possible next steps:
'''

value_prompt = '''Given the following math problem and the current solution steps, how likely is it that the steps will lead to the correct answer?

{input}

Current steps:
{steps}

Respond with one of: impossible, unlikely, possible, likely, sure.
'''

value_last_step_prompt = '''Given the following math problem, solution steps, and provided answer, how likely is it that the answer is correct?

{input}

Current steps:
{steps}

Answer:
{answer}

Respond with one of: impossible, unlikely, possible, likely, sure.
'''