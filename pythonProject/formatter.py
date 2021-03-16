file ='LIFE_wall.txt'

uniqlines = set(open(file,'r', encoding='windows-1251').readlines())
gotovo = open(file,'w', encoding='utf-8').writelines(set(uniqlines))