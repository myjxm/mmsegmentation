import csv
import os

headers = ['id','user_name','phone','email','email_password','email-token','password','autheticator','token',]

rows = []

fo = open("/Users/ddt/Documents/block-chain/discord账号/100.txt", "r")
id=1
for line in fo:
    li = line.split('----')
    rows.append([str(id),'','',li[0],li[1],li[2],li[3],li[4],li[5]])
    id = id+1
fo.close()
with open('/Users/ddt/Documents/block-chain/discord账号/100.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)
f.close()