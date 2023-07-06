import os
import shutil
pathsrc = '/Users/ddt/Documents/block-chain/xinjiang/新疆特集2019'
pathdes = '/Users/ddt/Documents/block-chain/xinjiang/闫慧云-part4'
pathfind = '/Users/ddt/Documents/block-chain/xinjiang/闫慧云-part1'
index=826
find_file=[]
for file in os.listdir(pathfind):
     find_file.append(file.replace('jpg',''))

for file_path  in os.listdir(pathsrc):
    if os.path.isfile(os.path.join(pathsrc, file_path)) == True:
        oldname = file_path.replace('jpg','').replace('heic','')
        print(oldname)
        if oldname not in find_file:
            shutil.copyfile(os.path.join(pathsrc, file_path), os.path.join(pathdes, file_path))