import random
import numpy as np
import torch
from transformers.utils import is_rich_available
import pandas as pd
import random

from huggingface_hub import login
import os

if is_rich_available():
    from rich.console import Console
    from rich.table import Table

def print_rich_table(df: pd.DataFrame) -> None:
    if not is_rich_available():
        raise ImportError(
            "The function `print_rich_table` requires the `rich` library. Please install it with `pip install rich`."
        )
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)

SYSTEM_PROMPT: str = "System: \nYou are a helpful assistant. Keep your responses concise and do not act as the user\n User:\n"
ASSISTANT_PREAMBLE: str = "\nAssistant:\n"

# TODO this isn't super safe
# ideally the LLM would have an API that would take in a struct that contained the necessary strings
# and would format as needed at the callsite. And the reward model would do the same.
def format_prompt(user_prompt: str, helpful_blunder: str = "") -> str:
    return SYSTEM_PROMPT + user_prompt + ASSISTANT_PREAMBLE + helpful_blunder

def remove_prompt_formatting(formatted_prompt: str) -> str:
    assert formatted_prompt.find(SYSTEM_PROMPT) == 0
    formatted_prompt = formatted_prompt[len(SYSTEM_PROMPT):]
    
    preamble_start_index = formatted_prompt.find(ASSISTANT_PREAMBLE)
    assert preamble_start_index >= 0

    return formatted_prompt[0:preamble_start_index]

def set_seed(seed):
    random.seed(seed)                     
    np.random.seed(seed)                  
    torch.manual_seed(seed)               
    torch.cuda.manual_seed(seed)          
    torch.cuda.manual_seed_all(seed)      

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

def login_to_hugging_face():
# if running this in Google Colab you'll need to add all of the files in this repo and run 
# pip install trl
# then you can paste this into a cell and run it

    try:
        # if using google colab you can add your token as a secret
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except:
        # if not then add it as an environment variable
        token = os.getenv('HF_TOKEN')

    # You need to authenticate with HuggingFace to use this model. The approval process took a few minutes for me.
    login(token)

def random_sentence():
    templates = [
        "The {adj} {noun} {verb} over the {adj2} {noun2}.",
        "A {noun} always {verb} when it is {adj}?",
        "A {noun}! It's a fun way to: play {verb} when it is {adj}$",
    ]

    words = {
        "adj": ["quick", "lazy", "happy", "strangé"],
        "adj2": ["blue", "dark", "silent", "bright", "underwater"],
        "noun": ["fox", "dog", "robot", "wizard", "1823", "Sir Kenneth Clark"],
        "noun2": ["moon", "river", "castle", "forest", "Delacroix", "model-predictive-control", "control barrier functions"],
        "verb": ["jumps", "runs", "flies", "dreams", "sub-terranen"],
    }

    template = random.choice(templates)
    return template.format(
        adj=random.choice(words["adj"]),
        adj2=random.choice(words["adj2"]),
        noun=random.choice(words["noun"]),
        noun2=random.choice(words["noun2"]),
        verb=random.choice(words["verb"]),
    )