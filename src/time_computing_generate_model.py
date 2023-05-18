import time
import subprocess
import os
from loguru import logger


# ROOT = src/

# Dans ./Instances

# Création d'un dossier Solutions

# Dans ce dossier Solutions : inst-xxx.lp

# Check si inst-xxx.sol existe déjà dans Solutions

file.replace('.txt', '.lp')

# Pour `gpsol` on va chercher inst-xxx.lp puis on le compile
#    `glpsol --lp inst-xxx.lp -o ./Solutions/inst-xxx.sol`



# Commande pour créer un fichier

# python3 linear_model_generation.py ./Instances/Instances/inst-xxx.txt















def time_computing(file):

    print(f'Running linear_model_generation.py on {file}')
    # try running linear_model_generation.py on file in under 5 minutes

    start = time.time()
    try:
        # via the command python3 chemin_augmentant.py inst-n-p.txt
        subprocess.run(['python3', 'linear_model_generation.py', file])

    except:
        logger.error('Python est introuvable.')
        logger.info('trying with python instead of python3')
        subprocess.run(['python3', 'linear_model_generation.py', file, 'timed'])

    end = time.time()
    elapsed_time = end - start
    logger.info(f'Elapsed time: {elapsed_time} seconds')

    if not os.path.exists("Timing"):
        os.makedirs("Timing")


    with open('Timing/' + file.replace('inst','').replace('.txt', '.path'), 'w') as f:
        f.write(f'Elapsed time: {end - start} seconds')

    with open('Timing/linear_all_in_seconds.txt', 'a') as f:
        f.write(f'{end - start} for {file} \n')

    # if elapsed_time under 5 minutes
    if elapsed_time < 300:
        with open('Timing/linear_all_in_seconds_under_5_mins.txt', 'a') as f:
            f.write(f'{end - start} for {file} \n')
    return elapsed_time

def compute_all_time():
    logger.info('Running linear_model_generation.py on all files')

    print(f'Current working directory: {os.getcwd()}')
    print(f"os.listdir(): {os.listdir().sort()}")
    print(f"os.listdir(): {os.listdir('.')}")

    # Attention, cela suppose que l'on est dans le dossier src
    # si on est dans le dossier parent :  ./src/Instances
    files = os.listdir('./Instances')
    files.sort()

    # Pour ne pas que le programme tourne trop longtemps
    elapsed_time = 0

    for file in files:
        if file.startswith('inst') and file.endswith('.txt') and elapsed_time < 120:
            elapsed_time += time_computing(file)
            logger.info(f'elapsed_time: {elapsed_time} seconds')


if __name__ == '__main__':
    compute_all_time()

