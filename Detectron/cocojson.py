import os
import json
import pandas as pd
from skimage.io import imread
import numpy as np
import cv2
images = []
annotations = []
categories = []
img_id = 0
ann_id = 0
detectron_dir = r'C:\Users\ADMIN\Desktop\Detectron'
dataset_dir = os.path.join(detectron_dir, 'nucleus')

images_dir = os.path.join(dataset_dir, 'val/images') 
anns_dir = os.path.join(dataset_dir, 'val/annots')

# Mapping of NuCLS class ids to COCO category ids
cat_mapping = {
   "tumor": 0,
  "fibroblast": 1, 
  "lymphocyte": 2,
  'plasma_cell': 3,
            'macrophage': 4,
            'mitotic_figure': 5,
            'vascular_endothelium': 6,
            'myoepithelium': 7,
            'apoptotic_body': 8,
            'neutrophil': 9,
            'ductal_epithelium': 10,
            'eosinophil': 11,
            'unlabeled': 12
}



for img_file in os.listdir(images_dir):
    img_path = os.path.join(images_dir, img_file)
    
    img_id += 1
    
    img = imread(img_path)
    height, width = img.shape[:2]
    
    images.append({
        "id": img_id,
        "width": width, 
        "height": height,
        "file_name": img_file
    })
    
    ann_df = pd.read_csv(os.path.join(anns_dir, 
                                      os.path.splitext(img_file)[0] + '.csv'))
    
    for _, ann_row in ann_df.iterrows():
        ann_type = ann_row['type']

        if ann_type == 'rectangle':

            # Rectangle handling  
            x1 = ann_row['xmin']  
            y1 = ann_row['ymin'] 
            x2 = ann_row['xmax']
            y2 = ann_row['ymax']
            w = x2 - x1
            h = y2 - y1
            segmentation = [[x1, y1], [x1, y1+h], [x1+w, y1+h], [x1+w, y1]]
            bbox = [x1, y1, x2-x1, y2-y1] 

        elif ann_type == 'polyline':
            vertex_x = ann_row['coords_x'].split(',')
            vertex_y = ann_row['coords_y'].split(',')
            
            segmentation = []
            for x, y in zip(vertex_x, vertex_y): 
                segmentation.append([int(x), int(y)])

            x, y, w, h = cv2.boundingRect(np.array(segmentation))
             
            
            
            bbox = [x, y, w, h]
        
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": cat_mapping[ann_row['raw_classification']], 
            "segmentation": segmentation,
            "area": bbox[2]*bbox[3],
            "bbox": bbox,
            "iscrowd": 0
        })
        
        ann_id += 1
        
categories = [{"id": v, "name": k} for k, v in cat_mapping.items()]
        
coco_format = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open('nuc_coco_val.json', 'w') as f:
    json.dump(coco_format, f)