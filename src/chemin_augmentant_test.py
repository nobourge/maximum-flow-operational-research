# runs chemin_augmentant.py on every instance
# via the command python3 chemin_augmentant.py inst-n-p.txt
# must generate a file model-n-p.path.
#
import os
import subprocess
import time

from loguru import logger

def main():
    logger.info('Running chemin_augmentant.py on every instance')
    #
    print(f'Current working directory: {os.getcwd()}')
    print(f"os.listdir(): {os.listdir().sort()}")
    print(f"os.listdir(): {os.listdir('.')}")
    print(f"os.listdir(): {os.listdir('Instances/Instances')}")

    # sort instances by complexity (number of nodes * density)
    listdir = os.listdir('.')
    for file in listdir:
        if file.startswith('inst-'):
            print(f'file: {file}')
    # for file in os.listdir("Instances/Instances"):


    # if "Path" directory does not exist, create it
    if not os.path.exists("Path"):
        os.makedirs("Path")

    if not os.path.exists("inst-100-0.1.txt"):
        logger.error('inst-100-0.1.txt does not exist')
        return
    else:
        logger.info('inst-100-0.1.txt exists at location: ' + os.path.abspath("inst-100-0.1.txt"))
    if not os.path.exists("inst-2-0.25.txt"):
        logger.error('inst-2-0.25.txt does not exist')
        return
    else:
        logger.info('inst-2-0.25.txt exists at location: ' + os.path.abspath("inst-2-0.25.txt"))
    # for file in current directory
    i = 0
    for file in os.listdir("."):
        i += 1
        logger.debug(f'i: {i}')
        logger.info(f'file: {file}')
    # for file in os.listdir("Instances/Instances"):
        # os.listdir() returns a list containing the names of the entries in the directory given by path.
        if file.startswith('inst-'):

            if os.path.exists("Path/"   + file.replace('inst','model')
                                              .replace('.txt',
                                                      '.path')) and \
               os.path.exists("Timing/" + file.replace('inst','model')
                                              .replace('.txt', '.path')):
                logger.info(f'Skipping {file}')
                continue
            print(f'Running chemin_augmentant.py on {file}')
            # try running chemin_augmentant.py on file in under 5 minutes

            start = time.time()
            try:
                # via the command python3 chemin_augmentant.py inst-n-p.txt
                # subprocess.run(['python3', 'chemin_augmentant.py', file])
                subprocess.run(['python', 'chemin_augmentant.py',
                                file, 'timed'])

            except:
                logger.error('Python est introuvable.')
                logger.info('trying with python instead of python3')
                subprocess.run(['python', 'chemin_augmentant.py', file, 'timed'])

            end = time.time()
            elapsed_time = end - start
            logger.info(f'Elapsed time: {elapsed_time} seconds')

            if not os.path.exists("Timing"):
                os.makedirs("Timing")
            with open('Timing/' + file.replace('inst',
                                             'model').replace('.txt', '.path'), 'w') as f:
                f.write(f'Elapsed time: {end - start} seconds')
            with open('Timing/all_in_seconds.txt', 'a') as f:
                f.write(f'{end - start} for {file} \n')
            #if elapsed_time under 5 minutes
            if elapsed_time < 300:
                with open('Timing/all_in_seconds_under_5_minutes.txt', 'a') as f:
                    f.write(f'{end - start} for {file} \n')



if __name__ == '__main__':
    main()
