# %%
import os
import cv2
import copy
import time
import torch
import numpy as np
import skimage
from cellpose import models
import matplotlib.pyplot as plt

# %%
def draw_nuclei(img, nuclei):
    overlay = copy.deepcopy(img)
    for nucleus in nuclei:
        x, y = nucleus
        cv2.circle(overlay, (x,y), 3, (255,0,0), -1)
    return overlay

# %%
def get_ihc_channel(ihc_rgb):
    ihc_hed = skimage.color.rgb2hed(ihc_rgb)
    ihc_h = ihc_hed[:, :, 0]
    ihc_dab = ihc_hed[:, :, 2]
    
    null_channal = np.zeros_like(ihc_h)
    ihc_h_rgb = skimage.color.hed2rgb(np.stack((ihc_h, null_channal, null_channal), axis=-1))
    ihc_h_rgb = np.array(ihc_h_rgb * 255, dtype=np.uint8)
    
    ihc_dab_rgb = skimage.color.hed2rgb(np.stack((null_channal, null_channal, ihc_dab), axis=-1))
    ihc_dab_rgb = np.array(ihc_dab_rgb * 255, dtype=np.uint8)
    
    return ihc_h, ihc_h_rgb, ihc_dab, ihc_dab_rgb

# %%
def mask2nuclei(mask):
    labels = skimage.measure.label(mask)
    props = skimage.measure.regionprops(labels)
    nuclei = []
    for prop in props:
        centroid_x, centroid_y = int(prop.centroid[0]), int(prop.centroid[1])
        nuclei.append((centroid_y, centroid_x))
    return nuclei

# %%
def get_nuclei_map(ihc_rgb, nuclei):
    ihc_nuclei_map = np.zeros(ihc_rgb.shape[:2])
    for nucleus in nuclei:
        centroid_y, centroid_x = nucleus
        ihc_nuclei_map[centroid_x][centroid_y] += 1
    ihc_nuclei_map = skimage.filters.gaussian(ihc_nuclei_map, sigma=11)  # old: 5, new: 11
    # ihc_nuclei_map = ihc_nuclei_map / np.max(ihc_nuclei_map)
    ihc_nuclei_map = (ihc_nuclei_map - np.min(ihc_nuclei_map)) / (np.max(ihc_nuclei_map) - np.min(ihc_nuclei_map))
    ihc_nuclei_map = np.array(ihc_nuclei_map * 255, dtype=np.uint8)
    return ihc_nuclei_map

# %%
def get_h_nuclei(ihc_h):
    ihc_h_seg = skimage.exposure.rescale_intensity(ihc_h, out_range=(0, 255), in_range=(np.min(ihc_h), np.percentile(ihc_h, 99)))
    threshold = skimage.filters.threshold_otsu(ihc_h_seg)
    
    ihc_h_seg[ihc_h_seg > threshold] = 255
    ihc_h_seg[ihc_h_seg <= threshold] = 0
    ihc_h_seg = np.array(ihc_h_seg, dtype=np.uint8)
    
    ihc_h_seg = (skimage.morphology.remove_small_holes(np.array(ihc_h_seg/255, bool), area_threshold=400, connectivity=2) * 255).astype(np.uint8)
    disk = skimage.morphology.disk(1)
    ihc_h_seg = skimage.morphology.dilation(ihc_h_seg, disk).astype(np.uint8)
    
    model = models.Cellpose(gpu=True, model_type="nuclei", net_avg=True, device=torch.device('cuda:0'))
    ihc_h_mask, _, _, _ = model.eval(ihc_h_seg, channels=[0,0])
    
    ihc_h_nuclei = mask2nuclei(ihc_h_mask)
    
    ihc_h_mask = skimage.color.label2rgb(ihc_h_mask, bg_label=0, bg_color=(0,0,0))
    ihc_h_mask = np.array(ihc_h_mask * 255, dtype=np.uint8)
        
    return ihc_h_seg, ihc_h_mask, ihc_h_nuclei

# %%
# def get_dab_nuclei(ihc_dab_rgb):
#     ihc_dab_hsv = skimage.color.rgb2hsv(ihc_dab_rgb / 255.0)
#     ihc_dab_s = ihc_dab_hsv[:, :, 1]
    
#     threshold = 0.15
#     ihc_dab_seg_mask = copy.deepcopy(ihc_dab_s)
#     ihc_dab_seg_mask[ihc_dab_seg_mask > threshold] = 255
#     ihc_dab_seg_mask[ihc_dab_seg_mask <= threshold] = 0

#     ihc_dab_seg_mask = np.array(ihc_dab_seg_mask, dtype=np.uint8)
    
#     model = models.Cellpose(gpu=True, model_type="cyto2", net_avg=True, device=torch.device('cuda:0'))
#     ihc_dab_mask, _, _, _ = model.eval(ihc_dab_seg_mask)
    
#     ihc_dab_mask = skimage.color.label2rgb(ihc_dab_mask, bg_label=0, bg_color=(0,0,0))
#     ihc_dab_mask = np.array(ihc_dab_mask * 255, dtype=np.uint8)
    
#     ihc_dab_nuclei = mask2nuclei(ihc_dab_mask)
    
#     return ihc_dab_mask, ihc_dab_nuclei

# %%
def get_dab_nuclei(ihc_dab_seg_mask):
    model = models.Cellpose(gpu=True, model_type="cyto2", net_avg=True, device=torch.device('cuda:0'))
    ihc_dab_mask, _, _, _ = model.eval(ihc_dab_seg_mask)
    
    ihc_dab_nuclei = mask2nuclei(ihc_dab_mask)
    
    ihc_dab_mask = skimage.color.label2rgb(ihc_dab_mask, bg_label=0, bg_color=(0,0,0))
    ihc_dab_mask = np.array(ihc_dab_mask * 255, dtype=np.uint8)
    
    return ihc_dab_mask, ihc_dab_nuclei

# %%
ihc_dir = "[DATASET DIR]/train_IHC"
dab_mask_dir = "[DATASET DIR]/train_IHC_dab_mask"

# %%
map_save_dir = "[DATASET DIR]/train_IHC_nuclei_map"
os.makedirs(map_save_dir, exist_ok=True)
overlay_save_dir = "[DATASET DIR]/train_IHC_overlay"
os.makedirs(overlay_save_dir, exist_ok=True)

# %%
img_list = os.listdir(ihc_dir)
img_list.sort()

for i in range(len(img_list)):
    time_s = time.time()
    
    img_id = img_list[i]
    ihc_bgr = cv2.imread(os.path.join(ihc_dir, img_id))
    ihc_rgb = cv2.cvtColor(ihc_bgr, cv2.COLOR_BGR2RGB)
    ihc_dab_seg_mask = cv2.imread(os.path.join(dab_mask_dir, img_id), 0)
    
    ihc_h, ihc_h_rgb, ihc_dab, ihc_dab_rgb = get_ihc_channel(ihc_rgb)
    
    ihc_h_seg, ihc_h_mask, ihc_h_nuclei = get_h_nuclei(ihc_h)
    ihc_h_overlay = draw_nuclei(ihc_h_rgb, ihc_h_nuclei)
    
    ihc_dab_mask, ihc_dab_nuclei = get_dab_nuclei(ihc_dab_seg_mask)
    ihc_dab_overlay = draw_nuclei(ihc_dab_rgb, ihc_dab_nuclei)
    
    ihc_nuclei = ihc_h_nuclei + ihc_dab_nuclei
    ihc_overlay = draw_nuclei(ihc_rgb, ihc_nuclei)
    ihc_nuclei_map = get_nuclei_map(ihc_rgb, ihc_nuclei)
    
    cv2.imwrite(os.path.join(map_save_dir, img_id), ihc_nuclei_map)
    cv2.imwrite(os.path.join(overlay_save_dir, img_id), cv2.cvtColor(ihc_overlay, cv2.COLOR_RGB2BGR))
    
    time_e = time.time()
    
    print("[{}/{} iter | time: {} s]---{} has been processed!".format(i+1,len(img_list),time_e-time_s,img_id))
    
    # fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, dpi=300)
    # ax = axes.ravel()
    # ax[0].imshow(ihc_rgb)
    # ax[1].imshow(ihc_overlay)
    # ax[2].imshow(ihc_nuclei_map, cmap='jet', vmin=0, vmax=255)
    # plt.show()
    # plt.close()
    
    # ihc_h_seg_vis = np.repeat(ihc_h_seg[:, :, np.newaxis], 3, axis=2)
    # ihc_dab_seg_mask_vis = np.repeat(ihc_dab_seg_mask[:, :, np.newaxis], 3, axis=2)
    # ihc_nuclei_map_vis = np.repeat(ihc_nuclei_map[:, :, np.newaxis], 3, axis=2)
    # fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, dpi=300)
    # ax = axes.ravel()
    # ax[0].imshow(ihc_rgb)
    # ax[1].imshow(ihc_h_rgb)
    # ax[2].imshow(ihc_h_mask)
    # ax[3].imshow(ihc_h_overlay)
    # ax[4].imshow(ihc_dab_rgb)
    # ax[5].imshow(ihc_dab_seg_mask_vis)
    # ax[6].imshow(ihc_dab_mask)
    # ax[7].imshow(ihc_overlay)
    # plt.show()
    # plt.close()


