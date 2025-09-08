import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data set path

DATA_PATH = os.path.join(BASE_DIR, '../data/demo')
# Result Path
RESULT_PATH = os.path.join(BASE_DIR, 'simulation')

# OpenAI Configuration
OPENAI_API_KEY = ''

# Simulation Parameters
SIM_PARAMS = {
    'memory_source':    'real',
    'learning_effect':  'yes',
    'forgetting_effect':'yes',
    'reflection_choice':'yes',
    'sim_strategy':     'performance',
    'gpt_type':         0,  # 0: GPT-3.5, 1: GPT-4
    'short_term_size':  5,
    'long_term_thresh': 5,
    'forget_lambda':    0.99
}
