import os

path = '/Users/ddt/Documents/block-chain/xinjiang/上传-part8'
index=0
list1 = [4,12,14,17,42,52,99,115,143,147,149,164,169,184,257,299,347,391,452,460,525,547,630,640,712,718,746,753,771,826,836,841,842,882,885,888,892,896,895,898,905,901,909,928,937,950]

list2 = [13,26,60,784,795,949,955,956,962,973]
list3=[119,129,926,847,849,285,282,258,73,682,566,574,861,864,894,877,264,834,916,874]


for file_path  in os.listdir(path):
    if os.path.isfile(os.path.join(path, file_path)) == True:
        oldname = os.path.join(path, file_path)
        if index < len(list3):
            file_name = '#' + str(list3[index]) + '.jpg'
            # 设置新文件名
            newname = os.path.join(path, file_name)

            os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
            print(oldname, '======>', newname)
            index = index+1