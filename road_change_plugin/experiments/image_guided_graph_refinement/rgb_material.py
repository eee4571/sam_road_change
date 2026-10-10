"""Cache-seeded RGB material graph cut. Classical image processing, no pretrained model."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from rasterio.features import rasterize
from skimage.morphology import remove_small_holes, remove_small_objects, disk


def reconstruct_area(image, prior):
    lo, hi = np.quantile(image.rgb[image.valid], [.02, .98], axis=0)
    rgb = np.clip((np.nan_to_num(image.rgb)-lo)/np.maximum(hi-lo, 1), 0, 1).astype('float32')
    smooth = cv2.bilateralFilter(rgb, 5, .07, 2)
    mean = cv2.GaussianBlur(smooth, (0, 0), 1/image.resolution)
    spread = np.sqrt(np.maximum(cv2.GaussianBlur(smooth**2, (0,0), 1/image.resolution)-mean**2,0).mean(axis=2))
    prior_mask = rasterize([(g, 1) for g in prior.geometry], out_shape=image.valid.shape,
                          transform=image.transform, all_touched=True).astype(bool)
    distance = distance_transform_edt(~prior_mask)*image.resolution
    r,g,b = smooth.transpose(2,0,1)
    green = 2*g-r-b
    brightness = smooth.mean(axis=2)
    color_ok = (green < .06) & (r-g < .25) & (b-r < .22) & (brightness < .70) & (brightness > .13)
    raw = cv2.GaussianBlur(np.nan_to_num(image.rgb), (0,0), .7/image.resolution)
    rr,gg,bb=raw.transpose(2,0,1)
    neutral=(rr-gg < 3.5) & (abs(gg-bb)<5) & (abs(rr-bb)<5)
    nonroad = (green > .10) | (r-g > .25) | (b-r > .25) | (brightness > .82)
    seeds = (distance < 4.) & color_ok & neutral & (spread < .04)
    mask = np.full(distance.shape, cv2.GC_PR_BGD, np.uint8)
    mask[(distance < 10) & color_ok & (r-g < .13)] = cv2.GC_PR_FGD
    mask[seeds] = cv2.GC_FGD
    mask[(distance > 22) | ~image.valid | nonroad] = cv2.GC_BGD
    if seeds.sum()<20:
        raise ValueError('Insufficient independent road-material seeds; no region was reconstructed')
    cv2.setRNGSeed(1729)
    cv2.grabCut(np.uint8(smooth*255), mask, None, np.zeros((1,65)), np.zeros((1,65)),
                5, cv2.GC_INIT_WITH_MASK)
    area = np.isin(mask, [cv2.GC_FGD,cv2.GC_PR_FGD]) & image.valid
    radius=max(1,int(round(1./image.resolution)))
    area=cv2.morphologyEx(area.astype('uint8'),cv2.MORPH_CLOSE,disk(radius).astype('uint8')).astype(bool)
    area=remove_small_holes(area,area_threshold=int(35/image.resolution**2))
    area=cv2.morphologyEx(area.astype('uint8'),cv2.MORPH_OPEN,disk(radius).astype('uint8')).astype(bool)
    area=remove_small_objects(area,min_size=int(50/image.resolution**2))
    labels,_=label(area)
    anchored=np.unique(labels[seeds])
    area &= np.isin(labels,anchored[anchored>0])
    return area,dict(rgb_material=mask,road_seed=seeds,prior_distance=distance,texture=spread)
