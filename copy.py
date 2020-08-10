import os
from tqdm import tqdm

annot_1 = open("./sreeram/annotations/s12020_07_30_20_09_38_483339.xml").readlines()
annot_2 = open("./sreeram/annotations/s12020_07_30_21_06_46_288497.xml").readlines()

ims = [i for i in os.listdir("./sreeram/") if i.split(".")[-1]=="jpg"]

ims_1 = [i for i in ims if i.split(".")[0].split("_")[3]=="20"]
ims_2 = [i for i in ims if i.split(".")[0].split("_")[3]=="21"]

for i in tqdm(ims_1):
    # print("./sreeram/annotations/"+i.split(".")[0]+".xml")
    f = open("./sreeram/annotations/"+i.split(".")[0]+".xml", "w")
    f.writelines(annot_1)
    f.close()

for i in tqdm(ims_2):
    # print("./sreeram/annotations/"+i.split(".")[0]+".xml")
    f = open("./sreeram/annotations/"+i.split(".")[0]+".xml", "w")
    f.writelines(annot_2)
    f.close()