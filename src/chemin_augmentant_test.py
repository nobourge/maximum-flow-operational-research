# runs chemin_augmentant.py on every instance
# via the command python3 chemin_augmentant.py inst-n-p.txt
# must generate a file model-n-p.path.
#
import os
import subprocess
import time

from loguru import logger

def main(algorithm, mode):

    logger.info('Running chemin_augmentant.py on every instance')
    #
    print(f'Current working directory: {os.getcwd()}')
    print(f"os.listdir(): {os.listdir().sort()}")
    print(f"os.listdir(): {os.listdir('.')}")
    #print(f"os.listdir(): {os.listdir('Instances')}")

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
#    for file in os.listdir("."):
#        i += 1
#        logger.debug(f'i: {i}')
#        logger.info(f'file: {file}')
    for file in os.listdir("Instances/Instances"):
        # os.listdir() returns a list containing the names of the entries in the directory given by path.
        if file.startswith('inst-'):
            file_model = file.replace('inst', 'model').replace('.txt', '.path')
            logger.info(f'file_model: {file_model}')
            # if file_model exists and is not empty
            # and file_model_timing exists
            # then skip

            if os.path.exists("Path/"+ file_model) and \
                os.path.exists(file_model) and \
                os.path.getsize(file_model) > 0 and \
                    os.path.exists("Timing/" + file_model):
                logger.info(f'Skipping {file}')
                continue
            print(f'Running chemin_augmentant.py on {file}')
            # try running chemin_augmentant.py on file in under 5 minutes

            start = time.time()
            try:
                if algorithm == 'Edmonds Karp':
                    if mode == 'debug':
                        # via the command python3 chemin_augmentant.py inst-n-p.txt
                        # subprocess.run(['python3', 'chemin_augmentant.py', file])
                        subprocess.run(['python', 'chemin_augmentant.py',
                                        file, 'timed'])
                elif algorithm == 'Distance Directed':
                    subprocess.run(['python', 'chemin_augmentant.py',
                                    file, 'distance_directed'])

            except:
                logger.error('Python est introuvable.')
                logger.info('trying with python instead of python3')
                subprocess.run(['python', 'chemin_augmentant.py', file, 'timed'])

            end = time.time()
            elapsed_time = end - start
            logger.info(f'Elapsed time: {elapsed_time} seconds')
            info= logger.info(f'Elapsed time: {elapsed_time} seconds')
            print("info: ", info)
            if algorithm == 'Edmonds Karp':
                path = 'Edmonds_Karp'
            elif algorithm == 'Dinic':
                path = 'Dinic'
            elif algorithm == 'Push Relabel':
                path = 'Push_Relabel'
            elif algorithm == 'Ford Fulkerson':
                path = 'Ford_Fulkerson'
            elif algorithm == 'Distance Directed':
                path = 'Distance_Directed'
            else:
                # logger.error(f'algorithm: {algorithm} is not valid')
                # return
                path = '.'
            save(file, elapsed_time, path)


def save(file, elapsed_time, path):
    Timing_directory = os.path.join(path, "Timing")
    if not os.path.exists(Timing_directory):
        os.makedirs(Timing_directory)
    with open(Timing_directory + file.replace('inst',
                                       'model').replace('.txt',
                                                        '.path'),
              'w') as f:
        f.write(f'Elapsed time: {elapsed_time} seconds')
    path_to_all_in_seconds_current = Timing_directory + \
                                     'all_in_seconds' \
                                     + time.strftime(
        "%Y%m%d-%H%M%S") + '.txt'
    with open(path_to_all_in_seconds_current, 'a') as f:
        f.write(f'{elapsed_time} for {file} \n')
    # if elapsed_time under 5 minutes
    if elapsed_time < 300:
        with open(Timing_directory +
                  'all_in_seconds_under_5_minutes.txt',
                  'a') as f:
            f.write(f'{elapsed_time} for {file} \n')


if __name__ == '__main__':
    # main('Edmonds Karp', 'debug')
    main('Distance Directed', 'debug')
