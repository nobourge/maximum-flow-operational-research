import time
import sys
import subprocess
import os
from loguru import logger
from linear_model_generation import solve_max_flow_glpk

def time_computing(file):

    print(f'Running linear_model_generation.py on {file}')
    # try running chemin_augmentant.py on file in under 5 minutes

    start = time.time()
    try:
        # via the command python3 chemin_augmentant.py inst-n-p.txt
        # subprocess.run(['python3', 'chemin_augmentant.py', file])
        subprocess.run(['python3', 'linear_model_generation.py', file, 'timed'])

    except:
        logger.error('Python est introuvable.')
        logger.info('trying with python instead of python3')
        subprocess.run(['python', 'linear_model_generation.py', file, 'timed'])

    end = time.time()
    elapsed_time = end - start
    logger.info(f'Elapsed time: {elapsed_time} seconds')

    if not os.path.exists("Timing"):
        os.makedirs("Timing")
    with open('Timing/' + file.replace('inst','model').replace('.txt', '.path'), 'w') as f:
        f.write(f'Elapsed time: {end - start} seconds')
    with open('Timing/linear_all_in_seconds.txt', 'a') as f:
        f.write(f'{end - start} for {file} \n')
    # if elapsed_time under 5 minutes
    if elapsed_time < 300:
        with open('Timing/linear_all_in_seconds_under_5_mins.txt', 'a') as f:
            f.write(f'{end - start} for {file} \n')

if __name__ == '__main__':
    file = sys.argv[1]
    time_computing(file)