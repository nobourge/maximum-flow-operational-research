# runs chemin_augmentant.py on every instance
# via the command python3 chemin_augmentant.py inst-n-p.txt
# must generate a origin_file model-n-p.path.
#
import os
import subprocess
import time

from loguru import logger


def get_instance_sorted_list():
    # sort instances by complexity (number of nodes * density)
    listdir = os.listdir('.')
    instance_sorted_list = []
    for file in listdir:
        if file.startswith('inst-'):
            # get number of nodes
            n = int(file.split('-')[1])
            # get density
            p = float(file.split('-')[2].replace('.txt', ''))
            # get complexity
            complexity = n * p
            instance_sorted_list.append((complexity, file))
    instance_sorted_list.sort()
    # logger.debug(f'instance_sorted_list: {instance_sorted_list}')
    return instance_sorted_list

def main(algorithm, mode=None):

    logger.info('Running chemin_augmentant.py on every instance')
    instance_sorted_list = get_instance_sorted_list()

    path = os.path.join(algorithm, "Timing")

    all_in_seconds = []
    for file in instance_sorted_list:
        file = file[1]
        file_model = file.replace('inst', 'model').replace('.txt', '.path')
        logger.info(f'file_model: {file_model}')

        logger.info(f'Running chemin_augmentant.py on {file}')
        # try running chemin_augmentant.py on origin_file in under 5 minutes

        start = time.time()
        try:
            subprocess.run(['python', 'chemin_augmentant.py',
                                    file, algorithm])
        except:
            logger.error('Python est introuvable.')
            logger.info('trying with python instead of python3')
            subprocess.run(['python', 'chemin_augmentant.py', file, algorithm])

        end = time.time()
        elapsed_time = end - start
        logger.info(f'Elapsed time: {elapsed_time} seconds')
        # info= logger.info(f'Elapsed time: {elapsed_time} seconds')
        # print("info: ", info)


        save(file, elapsed_time, path)
        all_in_seconds.append(f'{elapsed_time} for {file} \n')




def save(origin_file, elapsed_time, path):
    path += os.path.sep
    if not os.path.exists(path):
        os.makedirs(path)
    # with open(path + origin_file.replace('inst',
    #                                    'model').replace('.txt',
    #                                                     '.path'),
    #           'w') as f:
    #     f.write(f'Elapsed time: {elapsed_time} seconds')
    path_to_all_in_seconds_current = path + \
                                     'all_in_seconds' + '.txt'
    with open(path_to_all_in_seconds_current, 'a') as f:
        f.write(f'{elapsed_time} for {origin_file} \n')

    # if elapsed_time under 5 minutes
    if elapsed_time < 300:
        with open(path +
                  'all_in_seconds_under_5_minutes.txt',
                  'a') as f:
            f.write(f'{elapsed_time} for {origin_file} \n')


if __name__ == '__main__':
    main('edmonds_karp')
    # main('edmonds_karp', 'debug')
    # main('distance', 'debug')
