#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:21:25 2020

@author: ranjana
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
from tqdm import tqdm


import glob



#files1= glob.glob('D:\\video_template\\temp\\*.jpg')########### path for all template
files1= glob.glob('./temp_store3/*.jpg')
#files= glob.glob('D:\\video_template\\image\\*.jpg')##############  path for all images folder from video
files= glob.glob('./PMI_S3_15-09-2020-2/*.jpg')
path = './plot_15sep_1/'

total=pd.DataFrame()
#total1=pd.DataFrame()    

batch_size=200
total=pd.DataFrame()
total1=pd.DataFrame()
for n in range((len(files)//batch_size)+1):
    for file in tqdm(files[n*batch_size:(n+1)*batch_size]):
    #for file in files:
        image = io.imread(file,0)########## reading image
        #image = image[  16:500, 165:500] ################crop the image
        image = image[  0:350, 110:300]  ###################### for store3
        #plt.imshow(image)
        name1=file.split("/")[-1]
        name1=name1.split(".")[0]

        listTemplate = []
        for myfile in files1:
            temp0=io.imread(myfile)###########reading template
            name=myfile.split("/")[-1]
            name=name.split(".")[0]
            if ((name == "Stellar_var1") or(name == "Stellar_var2")): 
                name = "Stellar"
            if ((name == "Marlboro_gold_var1") or (name == "Marlboro_gold_var2")):
                name  = "Marlboro_gold"
            if name== "Marlboro_fusebeyond_var1":
                name = "Marlboro_fusebeyond"
            if name == "Stellar_define_var1":
                name ="Stellar_define"
            if name =="clove_mix_var1":
                name = "clove_mix"
            if ((name =="Wills_navycut_var1") or (name=="Wills_navycut_var2")):
                name = "Wills_navycut"
            if ((name =="Berkeley_var1") or (name =="Berkeley_var2") or
                (name =="Berkeley_var3")):
                name = "Berkeley"
            if ((name =="Mini_Light_var1") or (name =="Mini_Light_var2") or
                (name =="Mini_Light_var3")):
                name = "Mini_Light"
            if ((name =="Light_switch_var1") or (name =="Light_switch_var2") or
                (name =="Light_switch_var2")):
                name = "Light_switch"
           
            if ((name =="Marlboro_compact_var1") or (name =="Marlboro_compact_var2") or
               (name =="Marlboro_compact_var3") or (name =="Marlboro_compact_var4") or
               (name =="Marlboro_compact_var5") or (name =="Marlboro_compact_var6") or
               (name =="Marlboro_compact_var7") or (name =="Marlboro_compact_var8")):
                name = "Marlboro_compact"
            if ((name =="Flake_premium_var1") or(name =="Flake_premium_var2") or
                (name == "Flake_premium_var3")):
                name="Flake_premium"
            if ((name =="GoldFlake_Luxury King_var1") or(name =="GoldFlake_Luxury King_var2") or
                (name =="GoldFlake_Luxury King_var3") or (name == "GoldFlake_Luxury King_var4")):
                name = "GoldFlake_Luxury King"
            if ((name== "B&H_var1") or(name=="B&H_var2") or
                (name== "B&H_var3") or (name== "B&H_var4") or
                (name== "B&H_var5") or (name== "B&H_var6") or
                (name == "B&H_var7")):
                name = "B&H"
            if ((name== "Marlboro_adv_var1") or(name=="Marlboro_adv_var2") or
                (name== "Marlboro_adv_var3") or (name== "Marlboro_adv_var4") or
                (name== "Marlboro_adv_var5") or (name== "Marlboro_adv_var6")):
                name = "Marlboro_adv"
            if ((name== "Classic_mild_var1") or(name=="Classic_mild_var2") or
                (name== "Classic_mild_var3") or (name== "Classic_mild_var4") or
                (name== "Classic_mild_var5") or (name== "Classic_mild_var6")):
                name = "Classic_mild"
            if ((name== "Classic_menthol_var1") or(name=="Classic_menthol_var2") or
                (name== "Classic_menthol_var3") or (name== "Classic_menthol_var4") or
                (name== "Classic_menthol_var5") or (name== "Classic_menthol_var6")):
                name = "Classic_menthol"
            if ((name== "Classic_iceburst_var1") or(name=="Classic_iceburst_var2") or
                (name== "Classic_iceburst_var3") or (name== "Classic_iceburst_var4")):
                name = "Classic_iceburst"
            if ((name== "Classic_ultramild_var1") or(name=="Classic_ultramild_var2") or
                (name== "Classic_ultramild_var3") or (name== "Classic_ultramild_var4") or
                (name== "Classic_ultramild_var5") or (name== "Classic_ultramild_var6") or
                (name== "Classic_ultramild_var7") or (name== "Classic_ultramild_var8") or
                (name== "Classic_ultramild_var9") or (name== "Classic_ultramild_var10")):
                name = "Classic_ultramild"
            if ((name== "Classic_regular_var1") or(name=="Classic_regular_var2") or
                (name== "Classic_regular_var3") or (name== "Classic_regular_var4") or
                (name== "Classic_regular_var5")):
                name = "Classic_regular"
            if ((name== "GoldFlake_King_var1") or(name=="GoldFlake_King_var2") or
                (name== "GoldFlake_King_var3") or (name== "GoldFlake_King_var4") or
                (name== "GoldFlake_King_var5") or (name== "GoldFlake_King_var6") or
                (name== "GoldFlake_King_var7") or (name== "GoldFlake_King_var8") or
                (name == "GoldFlake_King_var9") or (name == "GoldFlake_King_var10")or 
                (name == "GoldFlake_King_var11") or (name == "GoldFlake_King_var12")or 
                (name == "GoldFlake_King_var13") or (name == "GoldFlake_King_var14") or
                (name == "GoldFlake_King_var15") or (name == "GoldFlake_King_var16") or
                (name == "GoldFlake_King_var17") or (name == "GoldFlake_King_var18")or
                (name == "GoldFlake_King_var19") or (name == "GoldFlake_King_var20") or
                (name == "GoldFlake_King_var21") or (name == "GoldFlake_King_var22") or
                (name == "GoldFlake_King_var23") or (name == "GoldFlake_King_var24") or
                (name == "GoldFlake_King_var25") or (name == "GoldFlake_King_var26")or
                (name == "GoldFlake_King_var27") or (name == "GoldFlake_King_var28") or
                (name == "GoldFlake_King_var29") or (name == "GoldFlake_King_var30") or
                (name == "GoldFlake_King_var31") or (name == "GoldFlake_King_var32") or
                (name == "GoldFlake_King_var33") or (name == "GoldFlake_King_var34")):
                name="GoldFlake_King"
            if ((name== "King_Light_var1") or(name=="King_Light_var2") or
                (name== "King_Light_var3") or (name== "King_Light_var4") or
                (name== "King_Light_var5") or (name== "King_Light_var6") or
                (name== "King_Light_var7") or (name== "King_Light_var8") or
                (name == "King_Light_var9") or (name == "King_Light_var10") or
                (name == "King_Light_var11") or (name == "King_Light_var12") or
                (name == "King_Light_var13") or (name == "King_Light_var14") or
                (name == "King_Light_var15") or (name == "King_Light_var16")):
                name="King_Light"

            #print("name of box",name)
            listTemplate.append( (name, temp0)) 

        for myfile in files1:
            temp0=io.imread(myfile)
            name=myfile.split("/")[-1]
            name=name.split(".")[0]
            if ((name == "Stellar_var1") or(name == "Stellar_var2")): 
                name = "Stellar"
            if ((name == "Marlboro_gold_var1") or (name == "Marlboro_gold_var2")):
                name  = "Marlboro_gold"
            if name== "Marlboro_fusebeyond_var1":
                name = "Marlboro_fusebeyond"
            if name == "Stellar_define_var1":
                name ="Stellar_define"
            if name =="clove_mix_var1":
                name = "clove_mix"
            if ((name =="Wills_navycut_var1") or (name=="Wills_navycut_var2")):
                name = "Wills_navycut"
            if ((name =="Berkeley_var1") or (name =="Berkeley_var2") or
                (name =="Berkeley_var3")):
                name = "Berkeley"
            if ((name =="Mini_Light_var1") or (name =="Mini_Light_var2") or
                (name =="Mini_Light_var3")):
                name = "Mini_Light"
            if ((name =="Light_switch_var1") or (name =="Light_switch_var2") or
                (name =="Light_switch_var2")):
                name = "Light_switch"
           
            if ((name =="Marlboro_compact_var1") or (name =="Marlboro_compact_var2") or
               (name =="Marlboro_compact_var3") or (name =="Marlboro_compact_var4") or
               (name =="Marlboro_compact_var5") or (name =="Marlboro_compact_var6") or
               (name =="Marlboro_compact_var7") or (name =="Marlboro_compact_var8")):
                name = "Marlboro_compact"
            if ((name =="Flake_premium_var1") or(name =="Flake_premium_var2") or
                (name == "Flake_premium_var3")):
                name="Flake_premium"
            if ((name =="GoldFlake_Luxury King_var1") or(name =="GoldFlake_Luxury King_var2") or
                (name =="GoldFlake_Luxury King_var3") or(name == "GoldFlake_Luxury King_var4")):
                name = "GoldFlake_Luxury King"
            if ((name== "B&H_var1") or(name=="B&H_var2") or
                (name== "B&H_var3") or (name== "B&H_var4") or
                (name== "B&H_var5") or (name== "B&H_var6") or
                (name == "B&H_var7")):
                name = "B&H"
            if ((name== "Marlboro_adv_var1") or(name=="Marlboro_adv_var2") or
                (name== "Marlboro_adv_var3") or (name== "Marlboro_adv_var4") or
                (name== "Marlboro_adv_var5") or (name== "Marlboro_adv_var6")):
                name = "Marlboro_adv"
            if ((name== "Classic_mild_var1") or(name=="Classic_mild_var2") or
                (name== "Classic_mild_var3") or (name== "Classic_mild_var4") or
                (name== "Classic_mild_var5") or (name== "Classic_mild_var6")):
                name = "Classic_mild"
            if ((name== "Classic_menthol_var1") or(name=="Classic_menthol_var2") or
                (name== "Classic_menthol_var3") or (name== "Classic_menthol_var4") or
                (name== "Classic_menthol_var5") or (name== "Classic_menthol_var6")):
                name = "Classic_menthol"
            if ((name== "Classic_iceburst_var1") or(name=="Classic_iceburst_var2") or
                (name== "Classic_iceburst_var3") or (name== "Classic_iceburst_var4")):
                name = "Classic_iceburst"
            if ((name== "Classic_ultramild_var1") or(name=="Classic_ultramild_var2") or
                (name== "Classic_ultramild_var3") or (name== "Classic_ultramild_var4") or
                (name== "Classic_ultramild_var5") or (name== "Classic_ultramild_var6") or
                (name== "Classic_ultramild_var7") or (name== "Classic_ultramild_var8") or
                (name== "Classic_ultramild_var9") or (name== "Classic_ultramild_var10")):
                name = "Classic_ultramild"
            if ((name== "Classic_regular_var1") or(name=="Classic_regular_var2") or
                (name== "Classic_regular_var3") or (name== "Classic_regular_var4") or
                (name== "Classic_regular_var5")):
                name = "Classic_regular"
            if ((name== "GoldFlake_King_var1") or(name=="GoldFlake_King_var2") or
                (name== "GoldFlake_King_var3") or (name== "GoldFlake_King_var4") or
                (name== "GoldFlake_King_var5") or (name== "GoldFlake_King_var6") or
                (name== "GoldFlake_King_var7") or (name== "GoldFlake_King_var8") or
                (name == "GoldFlake_King_var9") or (name == "GoldFlake_King_var10")or 
                (name == "GoldFlake_King_var11") or (name == "GoldFlake_King_var12")or 
                (name == "GoldFlake_King_var13") or (name == "GoldFlake_King_var14") or
                (name == "GoldFlake_King_var15") or (name == "GoldFlake_King_var16") or
                (name == "GoldFlake_King_var17") or (name == "GoldFlake_King_var18")or
                (name == "GoldFlake_King_var19") or (name == "GoldFlake_King_var20") or
                (name == "GoldFlake_King_var21") or (name == "GoldFlake_King_var22") or
                (name == "GoldFlake_King_var23") or (name == "GoldFlake_King_var24") or
                (name == "GoldFlake_King_var25") or (name == "GoldFlake_King_var26")or
                (name == "GoldFlake_King_var27") or (name == "GoldFlake_King_var28") or
                (name == "GoldFlake_King_var29") or (name == "GoldFlake_King_var30") or
                (name == "GoldFlake_King_var31") or (name == "GoldFlake_King_var32") or
                (name == "GoldFlake_King_var33") or (name == "GoldFlake_King_var34")):
                name="GoldFlake_King"
            if ((name== "King_Light_var1") or(name=="King_Light_var2") or
                (name== "King_Light_var3") or (name== "King_Light_var4") or
                (name== "King_Light_var5") or (name== "King_Light_var6") or
                (name== "King_Light_var7") or (name== "King_Light_var8") or
                (name == "King_Light_var9") or (name == "King_Light_var10") or
                (name == "King_Light_var11") or (name == "King_Light_var12") or
                (name == "King_Light_var13") or (name == "King_Light_var14") or
                (name == "King_Light_var15") or (name == "King_Light_var16")):
                name="King_Light"

            #print("name of box",name)

           # for i,angle in enumerate([90,180]):
               # rotated = np.rot90(temp0, k=i+1) # NB: rotate not good here, turns into float!
               # listTemplate.append( (name, rotated ) )
        


        im=image
    #Hits = matchTemplates(listTemplate, im,N_object=34, score_threshold=0.5, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0.25)
    #print(Hits)

    # Generate gaussian distributed noise, the noise intensity is set by the level variable
        noise = np.empty_like(im, dtype="int8")
        level = 10
        cv2.randn(noise,(0),(level)) # Matrix element are 0 in average

        imageNoise = cv2.add(im,noise, dtype=cv2.CV_8U)

        Hits_Noise = matchTemplates(listTemplate, imageNoise,score_threshold=0.75, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0.2,searchBox=None)##########without using N_object

        #Hits_Noise = matchTemplates(listTemplate, imageNoise,N_object=20,score_threshold=0.001, method=cv2.TM_CCOEFF_NORMED, maxOverlap=.08,searchBox=None)

        H=Hits_Noise
        df=Hits_Noise.reset_index()


        w = df["BBox"].str[2]
        h = df["BBox"].str[3]
        df["width"] = round(w * 2.54 / 96,1)
        df["hight"] = round(h * 2.54 / 96,1)

    #score=max(df.Score)
        f1=df.TemplateName.unique()

        br1=df['TemplateName'].value_counts().reset_index()


#df.to_csv("output.csv",index=False)

        df3=pd.DataFrame(columns=["index","TemplateName","BBox","Score","width","hight"])

        for i in range(0,len(br1.TemplateName)):
            df1=df[df.TemplateName ==f1[i]]
            score=round(max(df1.Score),2)
            score1=round((score-0.35),2)
            df2=df1[(df1.Score == score) | (df1.Score >= score1)]
            df3=df3.append(df2)
        df3["name"]=name1


    #df3.to_csv(name1+".csv",index=False)
    

    #Overlay1 = drawBoxesOnRGB(im, finalHits,showLabel=True,labelScale=0.4,labelColor=(0, 255, 255), boxThickness=2)

        Overlay2 = drawBoxesOnRGB(imageNoise,df3,showLabel=True,labelScale=0.2,labelColor=(0, 255, 255), boxThickness=1)###########draw the box on image

    #path=r'D:\video_template\plots'###############path to save image
    # plt.savefig(path+"\\"+name1+'.png')

        plt.figure(figsize = (20,20))
        plt.axis("off")
        plt.imshow(Overlay2)
        plt.savefig(path+"/"+name1+'.png')
        plt.close()

        br=df3[['TemplateName','name']]
        s2=br.pivot_table(index=['name'], columns=['TemplateName'], aggfunc=len).fillna(0).reset_index()
    #br2=br.groupby(["name",'TemplateName']).size().reset_index(name="Time")
    #br2=df3['TemplateName'].value_counts().reset_index()
    #br2.to_csv(name1+"_1.csv",index=False)
    #total1=s_t1.append(br2)
        total=total.append(s2)
        total.to_csv("all"+"_1.csv",index=False)
#total1.to_csv("all"+"_2.csv",index=False)
