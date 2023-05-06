# runs chemin_augmentant.py on every instance
# via the command python3 chemin_augmentant.py inst-n-p.txt
# must generate a file model-n-p.path.
#
import os
import subprocess

from loguru import logger

def main():
    logger.info('Running chemin_augmentant.py on every instance')
    #
    for file in os.listdir("Instances/Instances"):
        # os.listdir() returns a list containing the names of the entries in the directory given by path.
        if file.startswith('inst-'):
            print(f'Running chemin_augmentant.py on {file}')
            try:
                # via the command python3 chemin_augmentant.py inst-n-p.txt
                # subprocess.run(['python3', 'chemin_augmentant.py', file])
                subprocess.run(['python', 'chemin_augmentant.py', file])

            except:
                logger.error('Python est introuvable.')
                logger.info('trying with python instead of python3')
                subprocess.run(['python', 'chemin_augmentant.py', file])



if __name__ == '__main__':
    main()
