import os
import subprocess
import time
from loguru import logger

files = os.listdir('./Instances/Instances')
files.sort()

root = os.chdir('./')
cumulated_time = 0

last_file = ''

for file in files:

    file_path = os.abspath(f"Instances/Instances/{file}")

    while cumulated_time<300 :

        start = time.time()
        try :
            subprocess.run(["python3","model_generation.py", file_path])

        except :
            logger.info("Linear model not produced")

        end = time.time()
        cumulated_time += end

        elapsed_time = end - start
        logger.info(f"Elapsed time : {elapsed_time} seconds")

        if not os.path.exists("Timing"):
            os.makedirs("Timing")
        with open('Timing/' + file.replace('inst','model').replace('.txt', '.lp'), 'w') as f:
            f.write(f'Elapsed time: {end - start} seconds')

        with open('Timing/all_in_seconds.txt', 'a') as f:
            f.write(f'{end - start} for {file} \n')

    last_file = file

logger.info(f"Last lp file produced is {last_file}")

#solution_output = os.abspath(f"Instances/Solutions/{file.replace('txt', 'sol')}")
#output = subprocess.run(["glpsol", "--lp", file_path, "-o", solution_output])



