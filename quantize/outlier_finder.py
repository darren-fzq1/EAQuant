import numpy as np
import torch
import logging


def inter_category_distance(top_cate,bottom_cate):
    min_outlier = top_cate.min()
    max_outlier = bottom_cate.max()
    distance = (min_outlier-max_outlier).abs()
    return distance,min_outlier,max_outlier


def intra_category_variance(top_cate,bottom_cate):
    if top_cate.shape[0] == 1:
        top_variance = 0
    else:
        top_variance = torch.var(top_cate)
    if bottom_cate.shape[0] ==1:
        bottom_variance = 0
    else:
        bottom_variance = torch.var(bottom_cate)
    return top_variance,bottom_variance


def otsu(data,pos=True):
    data = data.sort(descending=True)[0]
    metric = -1000
    out_num = 0
    outlier = None
    outlier_queue =[]
    for idx,threshold in enumerate(data):
        up_cate = data[:idx+1]
        bottom_cate = data[idx+1:]
        if bottom_cate.shape[0] == 0 :
            continue 
        if pos == True:
            inter,min,max = inter_category_distance(up_cate,bottom_cate)
            _,bottom_intra = intra_category_variance(up_cate,bottom_cate)
            cur_metric = inter-bottom_intra
            
            if cur_metric > metric:
                metric = cur_metric
                out_num = idx+1
                outlier = data[:idx+1]
                outlier_threshold = bottom_cate[0]
        else:
            outlier_queue.append(data[idx])
            inter,min,max = inter_category_distance(up_cate,bottom_cate)
            top_intra,_ = intra_category_variance(up_cate,bottom_cate)
            cur_metric =  inter - top_intra 

            if cur_metric > metric:
                metric = cur_metric
                out_num = data.shape[0]-idx-1
                outlier = data[idx+1:]
                outlier_threshold = up_cate[-1]
    return out_num,outlier,outlier_threshold


