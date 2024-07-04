# %%
import os
import cv2
import copy
import time
import numpy as np
import skimage
import matplotlib.pyplot as plt

# %%
def get_dab_mask(ihc_rgb, if_blur=False):
    threshold = 0.15
    
    ihc_hed = skimage.color.rgb2hed(ihc_rgb)
    ihc_dab = ihc_hed[:, :, 2]
    
    null_channal = np.zeros_like(ihc_dab)
    ihc_dab_rgb = skimage.color.hed2rgb(np.stack((null_channal, null_channal, ihc_dab), axis=-1))
    
    ihc_dab_hsv = skimage.color.rgb2hsv(ihc_dab_rgb)
    ihc_dab_s = ihc_dab_hsv[:, :, 1]
    # print(np.max(ihc_dab_s), np.min(ihc_dab_s))
    
    ihc_dab_mask = copy.deepcopy(ihc_dab_s)
    ihc_dab_mask[ihc_dab_mask > threshold] = 255
    ihc_dab_mask[ihc_dab_mask <= threshold] = 0
    
    ihc_dab_rgb = np.array(ihc_dab_rgb * 255, dtype=np.uint8)
    ihc_dab_mask = np.array(ihc_dab_mask, dtype=np.uint8)
    
    if if_blur:
        ihc_dab_mask = cv2.GaussianBlur(ihc_dab_mask, (9,9), 3, 3)
    
    # for threshold in np.arange(0.1, 0.5+0.02, 0.02):
    #     ihc_dab_mask = copy.deepcopy(ihc_dab_s)
    #     ihc_dab_mask[ihc_dab_mask > threshold] = 255
    #     ihc_dab_mask[ihc_dab_mask <= threshold] = 0
    #     ihc_dab_mask = np.array(ihc_dab_mask, dtype=np.uint8)
    #     plt.imshow(ihc_dab_mask)
    #     plt.title(threshold)
    #     plt.show()
    #     plt.close()
    
    return ihc_dab_rgb, ihc_dab_mask


# %%
ihc_dir = "[DATASET DIR]/train_IHC"
dab_save_dir = "[DATASET DIR]/train_IHC_dab"
os.makedirs(dab_save_dir, exist_ok=True)
mask_save_dir = "[DATASET DIR]/train_IHC_dab_mask"
os.makedirs(mask_save_dir, exist_ok=True)

# %%
img_list = os.listdir(ihc_dir)
img_list.sort()

if_blur = True

for i in range(len(img_list)):
    time_s = time.time()
    
    img_id = img_list[i]
    ihc_bgr = cv2.imread(os.path.join(ihc_dir, img_id))
    ihc_rgb = cv2.cvtColor(ihc_bgr, cv2.COLOR_BGR2RGB)
    
    ihc_dab_rgb, ihc_dab_mask = get_dab_mask(ihc_rgb, if_blur)
    ihc_dab_bgr = cv2.cvtColor(ihc_dab_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(dab_save_dir, img_id), ihc_dab_bgr)
    cv2.imwrite(os.path.join(mask_save_dir, img_id), ihc_dab_mask)
    
    time_e = time.time()
    
    print("[{}/{} iter | time: {} s]---{} has been processed!".format(i+1,len(img_list),time_e-time_s,img_id))
    
    # ihc_dab_mask_vis = np.repeat(ihc_dab_mask[:, :, np.newaxis], 3, axis=2)
    # fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, dpi=200)
    # ax = axes.ravel()
    # ax[0].imshow(ihc_rgb)
    # ax[1].imshow(ihc_dab_rgb)
    # ax[2].imshow(ihc_dab_mask_vis)
    # plt.show()
    # plt.close()