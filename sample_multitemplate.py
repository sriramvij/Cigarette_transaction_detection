# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:41:01 2020

@author: Admin
"""


# 1st import the package and check its version
import MTM
print("MTM version : ", MTM.__version__)

from MTM import matchTemplates, drawBoxesOnRGB

import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MTM.NMS import NMS
import os
import glob
from tqdm import tqdm


files1= glob.glob('./temp_store3/*.jpg')########### all template
files= glob.glob('./PMI_S3_15-09-2020-2/*.jpg')##############all images folder from video
path='./new_stuff/'# Path to all new images

total=pd.DataFrame()
total1=pd.DataFrame()
for file in tqdm(files[:100]):
    listTemplate = []
    for myfile in files1:
        temp1=io.imread(myfile)###########reading template
        name=myfile.split("/")[-2]
        name=name.split(".")[-1]
        if name == "Stellar_var0":
            name = "Stellar"
        if name == "Marlboro_gold_var0":
            name  = "Marlboro_gold"
        if name== "Marlboro_fusebeyond_var0":
            name = "Marlboro_fusebeyond"
        if name == "Stellar_define_var0":
            name ="Stellar_define"
        if name =="clove_mix_var0":
            name = "clove_mix"
        if name =="Flake_premium_var0":
            name="Flake_premium"
        if ((name== "GoldFlake_King_var0") or (name=="GoldFlake_King_var2") or
            (name== "GoldFlake_King_var2") or (name== "GoldFlake_King_var4") or
            (name== "GoldFlake_King_var4") or (name== "GoldFlake_King_var6") or
            (name== "GoldFlake_King_var6") or (name== "GoldFlake_King_var8") or
            (name == "GoldFlake_King_var8")):
            name="GoldFlake_King"
        if ((name== "GoldFlake_King_Light_var0") or(name=="GoldFlake_King_Light_var2") or
            (name== "GoldFlake_King_Light_var2") or (name== "GoldFlake_King_Light_var4") or
            (name== "GoldFlake_King_Light_var4") or (name== "GoldFlake_King_Light_var6") or
            (name== "GoldFlake_King_Light_var6") or (name== "GoldFlake_King_Light_var8") or
            (name== "GoldFlake_King_Light_var8")):
            name="GoldFlake_King_Light"
        if name =="Clove_crush_var0":
            name="Clove_crush"
        if name == "B&H_var0":
            name = "B&H"
        if name == "Classic_iceburst_var0":
            name = "Classic_iceburst"
        if name == "Marlboro_adv_iceburst_var0":
            name = "Marlboro_adv"
    listTemplate.append((name, temp1)) 


    for myfile in files1:
        temp0=io.imread(myfile)
        name=myfile.split("/")[-2]
        name=name.split(".")[-1]
        if name == "Stellar_var0": 
            name = "Stellar"
        if name == "Marlboro_gold_var0":
            name  = "Marlboro_gold"
        if name== "Marlboro_fusebeyond_var0":
            name = "Marlboro_fusebeyond"
        if name == "Stellar_define_var0":
            name ="Stellar_define"
        if name =="clove_mix_var0":
            name = "clove_mix"
        if name =="Flake_premium_var0":
            name="Flake_premium"
        if ((name== "GoldFlake_King_var0") or(name=="GoldFlake_King_var2") or
            (name== "GoldFlake_King_var2") or (name== "GoldFlake_King_var4") or
            (name== "GoldFlake_King_var4") or (name== "GoldFlake_King_var6") or
            (name== "GoldFlake_King_var6") or (name== "GoldFlake_King_var8") or
            (name == "GoldFlake_King_var8")):
            name="GoldFlake_King"
        if ((name== "GoldFlake_King_Light_var0") or(name=="GoldFlake_King_Light_var2") or
            (name== "GoldFlake_King_Light_var2") or (name== "GoldFlake_King_Light_var4") or
            (name== "GoldFlake_King_Light_var4") or (name== "GoldFlake_King_Light_var6") or
            (name== "GoldFlake_King_Light_var6") or (name== "GoldFlake_King_Light_var8") or
            (name == "GoldFlake_King_Light_var8")):
            name="GoldFlake_King_Light"
        if name =="Clove_crush_var0":
            name="Clove_crush"
        if name == "B&H_var0":
            name = "B&H"
        if name == "Classic_iceburst_var0":
            name = "Classic_iceburst"
        if name == "Marlboro_adv_iceburst_var0":
            name = "Marlboro_adv"

        for i,angle in enumerate([90,180]):
            rotated = np.rot90(temp0, k=i+1) # NB: rotate not good here, turns into float!
            listTemplate.append( (name, rotated ) )
        try:
            image = io.imread(file,0)########## reading image
            image = image[ 0:500, 165:500] 
            name1=file.split("/")[-1]
            name1=name1.split(".")[0]

            im=image
            noise = np.empty_like(im, dtype="int8")
            level = 10
            cv2.randn(noise,(0),(level)) # Matrix element are 0 in average

            imageNoise = cv2.add(im,noise, dtype=cv2.CV_8U)



            Hits_Noise = matchTemplates(listTemplate, imageNoise,N_object=90,score_threshold=0.001, method=cv2.TM_CCOEFF_NORMED, maxOverlap=.08)

            H=Hits_Noise
            df=Hits_Noise.reset_index()


            w = df["BBox"].str[2]
            h = df["BBox"].str[3]
            df["width"] = round(w * 2.54 / 96,1)
            df["hight"] = round(h * 2.54 / 96,1)

            f1=df.TemplateName.unique()

            br1=df['TemplateName'].value_counts().reset_index()



            df3=pd.DataFrame(columns=["index","TemplateName","BBox","Score","width","hight"])

            for i in range(0,len(br1.TemplateName)):
                df1=df[df.TemplateName ==f1[i]]
                score=round(max(df1.Score),2)
                score1=round((score-0.35),2)
                df2=df1[(df1.Score == score) | (df1.Score >= score1)]
                df3=df3.append(df2)
            df3["name"]=name1

            Overlay2 = drawBoxesOnRGB(imageNoise,df3,showLabel=True,labelScale=0.4,labelColor=(0, 255, 255), boxThickness=2)

            plt.figure(figsize = (20,20))
            plt.axis("off")
            plt.imshow(Overlay2) 
            plt.savefig(path+name1+'.png')
            plt.close()

            br=df3[['TemplateName','name']]
            s2=br.pivot_table(index=['name'], columns=['TemplateName'], aggfunc=len).fillna(0).reset_index()
            br2=br.groupby(["name",'TemplateName']).size().reset_index(name="Time")
            total=total.append(s2)

        except:
            pass 
total.to_csv("all_2020_11_09"+"_1.csv",index=False)
