import os
import subprocess
import time
from loguru import logger
import re

import natsort

def sort_files(files):
    pattern = r"^inst-.+\.txt$"
    filtered_files = [file for file in files if re.match(pattern, file)]

    return filtered_files


def catch_instances():
    list_all_files = os.listdir('./Instances/Instances')

    pattern = r"^inst-.+\.txt$"
    filtered_files = [file for file in list_all_files if re.match(pattern, file)]

    # Pour trier les fichiers
    files = natsort.natsorted(filtered_files)

    return files

instances = catch_instances()

print(f"Instances = {instances}")

root = os.chdir('./')
cumulated_time = 0

last_file = ''

with open('Timing/linear_all_in_seconds.txt', 'w') as f:
    f.write('')

for instance in instances:

    if cumulated_time < 300:
        instance_path = os.path.abspath(f"Instances/Instances/{instance}")
        logger.info(f"Run model generation on {instance}")

        print(cumulated_time)
        start = time.time()

        try :
            subprocess.run(["python3","model_generation.py", instance_path])

        except :
            logger.info("Linear model not produced")

        end = time.time()
        elapsed_time = end - start
        cumulated_time += elapsed_time
        logger.info(f"Elapsed time : {elapsed_time} seconds")

        if not os.path.exists("Timing"):
            os.makedirs("Timing")
        with open('Timing/' + instance.replace('inst', 'model').replace('.txt', '.lp'), 'w') as f:
            f.write(f'Elapsed time: {end - start} seconds')

        with open('Timing/linear_all_in_seconds.txt', 'a') as f:
            f.write(f'{end - start} seconds for {instance} \n')

        last_file = instance
    else:
        logger.info(f"Time limit reached with {cumulated_time} seconds")


with open('Timing/linear_all_in_seconds.txt', 'a') as f:
    f.write(f"Last lp file produced is {last_file} with {cumulated_time} seconds \n")

logger.info(f"Last lp file produced is {last_file} with {cumulated_time} seconds")

#solution_output = os.abspath(f"Instances/Solutions/{file.replace('txt', 'sol')}")
#output = subprocess.run(["glpsol", "--lp", file_path, "-o", solution_output])



