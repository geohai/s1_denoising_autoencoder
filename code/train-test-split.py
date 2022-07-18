import os
import random

train_list = [f for f in os.listdir("../../s1_res_learn/data/train/noisy") \
              if f[-12:-8]=="8192"]

test_list = train_list[:6]
print(*test_list, sep="\n")

for file in test_list:
    os.system("cp ../../s1_res_learn/data/train/noisy/"+file+" "+\
              "../../s1_res_learn/data/test/noisy/"+file)
    os.system("cp ../../s1_res_learn/data/train/clean/"+file+" "+\
              "../../s1_res_learn/data/test/clean/"+file)

new_test_list = [f for f in os.listdir("../../s1_res_learn/data/test/noisy") \
                 if f[-12:-8]=="8192"]

print(*new_test_list, sep="\n")
for file in new_test_list:
    if os.path.isfile("../../s1_res_learn/data/test/noisy/"+file):
        os.remove("../../s1_res_learn/data/train/noisy/"+file)
        print("removed ../../s1_res_learn/data/train/noisy/"+file)
