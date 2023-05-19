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

def main(algorithm, mode=''):

    logger.info('Running chemin_augmentant.py on every instance')
    logger.info(f'algorithm: {algorithm}')
    instance_sorted_list = get_instance_sorted_list()

    # path = os.path.join(algorithm, "Timing")
    # all_in_seconds_path = os.path.join(path, 'all_in_seconds')
    # all_in_seconds_path += time.strftime("%Y|%m|%d-%H-%M-%S") + '.txt'
    # all_in_seconds_path = '{algorithm}, Timing",
    #                                    'all_in_seconds', time.strftime("%Y|%m|%d-%H-%M-%S") + '.txt')
    all_in_seconds_path = algorithm + os.path.sep \
                          +'Timing' + os.path.sep \
                          + 'all_in_seconds' + os.path.sep \
                          + time.strftime("%Y-%m-%d-%H-%M-%S") \
                          + '.txt'
    logger.info(f'all_in_seconds_path: {all_in_seconds_path}')
    if not os.path.exists(os.path.dirname(all_in_seconds_path)):
        os.makedirs(os.path.dirname(all_in_seconds_path))

    all_in_seconds = []
    # all_in_seconds_path = path + 'all_in_seconds.txt' + time.strftime(
    #         "%Y|%m|%d-%H-%M-%S") + '.txt'
    for _file in instance_sorted_list:
        _file = _file[1]
        file_model = _file.replace('inst', 'model').replace('.txt', '.path')
        logger.info(f'file_model: {file_model}')

        logger.info(f'Running chemin_augmentant.py on {_file}')
        # try running chemin_augmentant.py on origin_file in under 5 minutes

        start = time.time()
        try:
            subprocess.run(['python', 'chemin_augmentant.py',
                            _file, algorithm, mode])
        except:
            logger.error('Python est introuvable.')
            logger.info('trying with python instead of python3')
            subprocess.run(['python', 'chemin_augmentant.py', _file, algorithm, mode])
            raise

        end = time.time()
        elapsed_time = end - start
        logger.info(f'Elapsed time: {elapsed_time} seconds')

        # save(_file, elapsed_time, path)
        save(all_in_seconds_path, elapsed_time, _file)
    #     all_in_seconds.append(f'{elapsed_time} for {_file} \n')
    # with open(all_in_seconds_path), 'w') as f:
    #     f.writelines(all_in_seconds)

def save(path, elapsed_time, origin_file=None):
    #
    # path += os.path.sep
    # if not os.path.exists(path):
    #     os.makedirs(path)
    #
    # path_to_all_in_seconds_current = path + \
    #                                  'all_in_seconds' + '.txt'
    with open(path, 'a') as f:
        f.write(f'{elapsed_time} for {origin_file} \n')

    # if elapsed_time under 5 minutes
    # if elapsed_time < 300:
    #     with open(path +
    #               'all_in_seconds_under_5_minutes.txt',
    #               'a') as f:
    #         f.write(f'{elapsed_time} for {origin_file} \n')


if __name__ == '__main__':
    # main('edmonds_karp')
    # main('edmonds_karp', 'debug')
    main('relabel_to_front')
    # main('distance', 'debug')
