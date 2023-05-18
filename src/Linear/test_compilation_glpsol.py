import subprocess
import os


def unit_compilation(file):
    #file = 'inst-100-0.2.lp'
    #    `glpsol --lp inst-xxx.lp -o ./Solutions/inst-xxx.sol`
    file = file.replace('.txt', '.lp')
    os.chdir('../..')
    os.chdir('Instances/Solutions')

    if os.path.exists(file.replace('.lp', '.sol')):
        os.remove(file.replace('.lp', '.sol'))

    output = subprocess.run(["glpsol", "--lp", file, "-o", file.replace('.lp', '.sol')])

    solution = subprocess.run(["head",  f"{file.replace('.lp', '.sol')}"])

    print(solution)

if __name__ == '__main__':
    os.chdir('Instances/Solutions')
    unit_compilation('inst-100-0.1.lp')
