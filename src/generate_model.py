# script python3 nomm´e generate model.py
# prenant en param`etre en ligne de commande le nom
# d’une instance inst-n-p.txt dans le mˆeme dossier
# et qui g´en`ere un programme lin´eaire en nombre
# entiers de cette instance au format CPLEX LP vu en TP.
# Ce programme doit ˆetre sauv´e dans un fichier
# model-n-p.lp. Le script appel´e sur l’instance inst-300-0.3.txt via la commande
# python3 generate model.py inst-300-0.3.txt
# doit g´en´erer un ficher model-300-0.3.lp.
# Comme utilis´e en TP, le fichier doit pouvoir ˆetre r´esolu et
# sauver les r´esultats avec la commande
# glpsol --lp model-300-0.3.lp -o model-300-0.3.sol
