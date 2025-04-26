import re
import os
import sympy
import pandas as pd
import pdb

from tree_of_thoughts.prompts_mathvista import standard_prompt, cot_prompt, propose_prompt, value_prompt, value_last_step_prompt

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
LINE_DELIMITER = '\n'
ANSWER_PREFIX = "answer:"
USER_PROMPT_SOLN_PREFIX = "Solution:"

def _preprocess_user_prompt(user_prompt: str) -> str:
    # Preprocess user_prompt
    soln_idx = user_prompt.find(USER_PROMPT_SOLN_PREFIX)
    if soln_idx == -1:
        raise(ValueError("Solution prefix not found within user_prompt."))
    user_prompt = user_prompt[:soln_idx]
    return user_prompt

class MathVistaTask(object):
    """
    Input (x)   : MathVista problem
    Output (y)  : Steps (max N) to reach the final result
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: @TODO
    Output Example: @TODO
    """
    def __init__(self, max_depth = 8):
        self.max_depth = max_depth

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        x = _preprocess_user_prompt(x)
        return standard_prompt.format(input=x)

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        x = _preprocess_user_prompt(x)
        return cot_prompt.format(input=x, steps=y) # @hlwong: verify

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        """
        Use value_last_step_prompt if the last line is the answer, otherwise use value_prompt.
        """
        x = _preprocess_user_prompt(x)
        lines = y.strip().split(LINE_DELIMITER)
        last_line = lines[-1]
        if last_line.strip().lower().startswith(ANSWER_PREFIX):
            ans = last_line[len(ANSWER_PREFIX):].strip()
            steps = LINE_DELIMITER.join(lines[:-1])
            return value_last_step_prompt.format(input=x, steps=steps, answer=ans)
        return value_prompt.format(input=x, steps=y)

    @staticmethod
    def value_outputs_unwrap(y: str, value_output: str) -> float:
        """
        Map the model's output to a numeric value for MathVista.
        """
        pdb.set_trace()
        value_name = str(value_output).strip().lower().split(LINE_DELIMITER)[-1]
        value_map = {
            'impossible': 0.001,
            'unlikely': 0.1,
            'possible': 0.5,
            'likely': 1,
            'sure': 2
        }
        value = value_map.get(value_name, 0)
        # value = sum(value_map.get(name, 0) for name in value_names) # @hlwong:
        return value