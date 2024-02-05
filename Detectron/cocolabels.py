

import os
import random
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches 
import cv2
import numpy as np
import matplotlib.path as mpath
def display_images_with_coco_annotations(image_paths, annotations, display_type='both'):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    for ax, img_path in zip(axs.ravel(), image_paths):
        # Load image using OpenCV and convert it from BGR to RGB color space
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ax.imshow(image)
        ax.axis('off')  # Turn off the axes

        # Get image filename to match with annotations
        img_filename = os.path.basename(img_path)
        img_id = next(item for item in annotations['images'] if item["file_name"] == img_filename)['id']

        # Filter annotations for the current image
        img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]
        
        colors = [np.random.rand(3,) for _ in range(len(img_annotations))]  # Generate random colors
        
        for ann, color in zip(img_annotations, colors):
            # Display bounding box
            if display_type in ['bbox', 'both']:
                bbox = ann['bbox']
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                         linewidth=1, edgecolor=color, 
                                         facecolor='none')
                ax.add_patch(rect)
            
            # Display segmentation polygon
            if display_type in ['seg', 'both']:
                for ann in img_annotations:

                    poly_verts = ann['segmentation']
                    
                    # Ensure Nx2 array shape  
                    poly_verts = np.array(poly_verts)
                    if poly_verts.ndim == 1:
                        poly_verts = poly_verts.reshape(-1, 2)
                        
                    polygon = mpath.Path(poly_verts)
                    
                    patch = mpatches.PathPatch(polygon, facecolor='red', lw=2) 
                    ax.add_patch(patch)

        ax.set_title(img_filename, fontsize=10)

    plt.tight_layout()
    plt.show()

# Load COCO annotations
with open(r'C:\Users\ADMIN\Desktop\Detectron\nuc_coco.json', 'r') as f:
    annotations = json.load(f)

# Get all image files
image_dir = r"C:\Users\ADMIN\Desktop\Detectron\nucleus\images"
all_image_files = [os.path.join(image_dir, img['file_name']) for img in annotations['images']]
random_image_files = random.sample(all_image_files, 4)

# Choose between 'bbox', 'seg', or 'both'
display_type = 'seg'
display_images_with_coco_annotations(random_image_files, annotations, display_type)