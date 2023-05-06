# runs generate_model.py on every instance
# via the command python3 generate_model.py inst-n-p.txt
# must generate a file model-n-p.lp.

import os
import subprocess

def main():
    #
    for file in os.listdir("Instances/Instances"):
        # os.listdir() returns a list containing the names of the entries in the directory given by path.
        if file.startswith('inst-'):
            print(f'Running generate_model.py on {file}')
            subprocess.run(['python3', 'generate_model.py', os.path.join('test', file)])

if __name__ == '__main__':
    main()
