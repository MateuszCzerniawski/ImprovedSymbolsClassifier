import subprocess
import multiprocessing as mp

mp.set_start_method('spawn')

# print('nets pca')
# nets_pca = subprocess.run(['../.venv/Scripts/python.exe', 'neural.py', "__pca__"], text=True)
# print('nets bil')
# nets_bil = subprocess.run(['../.venv/Scripts/python.exe', 'neural.py', "__bilinear__"], text=True)
#
# print('trees pca')
# trees_pca = subprocess.run(['../.venv/Scripts/python.exe', 'trees.py', "__pca__"], text=True)
# print('trees bil')
# trees_bil = subprocess.run(['../.venv/Scripts/python.exe', 'trees.py', "__bilinear__"], text=True)

# print('cv nets pca')
# cv_nets_pca = subprocess.run(['../.venv/Scripts/python.exe', 'neural.py', "__cv-pca__"], text=True)
# print('cv nets bil')
# cv_nets_bil = subprocess.run(['../.venv/Scripts/python.exe', 'neural.py', "__cv-bil__"], text=True)

# print('cv trees pca')
# cv_trees_pca = subprocess.run(['../.venv/Scripts/python.exe', 'trees.py', "__cv-pca__"], text=True)
print('cv trees bil')
cv_trees_bil = subprocess.run(['../.venv/Scripts/python.exe', 'trees.py', "__cv-bil__"], text=True)

# print('conv')
# conv = subprocess.run(['../.venv/Scripts/python.exe', 'conv.py', "__conv__"], text=True)
