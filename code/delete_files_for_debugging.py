import os

dir = "/media/benjamin/blue_benny/data/train/"

raw_list = [f for f in os.listdir(dir+"input/")]

noise_vec_list = [f for f in os.listdir(dir+"noise_vec/")]
denoised_list = [f for f in os.listdir(dir+"denoised/")]

for list in [noise_vec_list, denoised_list]:
    difference = [f for f in list if f not in raw_list]

    if len(difference):
        for f in difference:
            if os.path.isfile(dir+"denoised/"+f):
                os.remove(dir+"denoised/"+f)
                print("deleted denoised")
            if os.path.isfile(dir+"noise_vec/"+f):
                os.remove(dir+"noise_vec/"+f)
                print("deleted noise vectors")

    else:
        print("no different files")

# import random
# import shutil
#
# random.shuffle(raw_list)
#
# raw_list_10 = raw_list[:10]
#
# test_dir = "/media/benjamin/blue_benny/data/test/"
# for f in raw_list_10:
#     shutil.move(dir+"input/"+f, test_dir+"input/"+f)
#     shutil.move(dir+"noise_vec/"+f, test_dir+"noise_vec/"+f)
#     shutil.move(dir+"denoised/"+f, test_dir+"denoised/"+f)
