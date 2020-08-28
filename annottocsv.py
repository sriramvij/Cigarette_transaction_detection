from tqdm import tqdm
import pandas as pd
import numpy as np
import json

fname = "store2.json"

annots = [json.loads(i) for i in open(fname).readlines()]

df = pd.DataFrame(columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']) 

print(df)

for json_data in tqdm(annots):
    im_name = json_data["content"].split("___")[-1]
    if json_data["annotation"]!=None:
        for i in json_data["annotation"]:
            obj_name = i["label"]
            points = np.array(i["points"])
            xmax = np.amax(points[:, 0])
            xmin = np.amin(points[:, 0])
            ymax = np.amax(points[:, 1])
            ymin = np.amin(points[:, 1])
            width = i["imageWidth"]
            height = i["imageHeight"]
            df = df.append({"filename":"_".join(im_name.split("_")[1:]),
                            "width":width,
                            "height":height,
                            "class":obj_name[0],
                            "xmin":int(xmin*width),
                            "ymin":int(ymin*height),
                            "xmax":int(xmax*width),
                            "ymax":int(ymax*height)
                            }, ignore_index = True)


df.to_csv(fname.split(".")[0]+".csv", index=False)