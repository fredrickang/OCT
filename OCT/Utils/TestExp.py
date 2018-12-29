from Utils.ExpressiveGradients import *
from Vis.VisualizationTool import *

import pandas as pd
from pandas import DataFrame as df
import cv2
import itertools


def ground_truth_box(GTimage):
    ''' for single Ground truth image'''
    box_point = []
    img_g = GTimage[:, :, 1]
    img_b = GTimage[:, :, 2]
    img_d = img_b - img_g
    thresh = cv2.threshold(img_d, 50, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    for (j, d) in enumerate(cnts):
        if cv2.contourArea(d) < 150:
            continue
        rect = cv2.minAreaRect(d)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box_point.append(box)

    return box_point


def get_location_map_rect(box_list, dim):
    mat_list = []
    for bx in box_list:
        mat = np.zeros(dim)
        top_left = np.clip(np.min(bx, axis=0), 0, 342)
        bot_right = np.clip(np.max(bx, axis=0), 0, 342)
        idx = list(itertools.product(range(top_left[1], bot_right[1]), range(top_left[0], bot_right[0])))
        for ids in idx:
            mat[ids[0], ids[1], :] = 1
        mat_list.append(mat)
    return mat_list


def GT_location_map(GTimage_list, dim):
    loc_map = []
    i = 0
    for img in tqdm(GTimage_list):
        box_list = ground_truth_box(img)
        mat_list = get_location_map_rect(box_list, dim)
        loc_map.append(sum(mat_list))
    return loc_map


def compare_GT_by_gt(raw_img, attr_img, GT_map, topK=100, min_att=0.3):
    # attribution의 map 도출
    attr_boxes = find_topK_rect(raw_img, attr_img, topK, min_att, mode=False)
    attr_bx_maps = get_location_map(attr_boxes, attr_img.shape)
    attr_map = np.clip(sum(attr_bx_maps), 0, 1)

    # GT의 map 계산
    gt_map = np.clip(GT_map, 0, 1)

    # 두개 더해 공통부분 추출
    inter_map = attr_map + gt_map
    # 공통 이미지 중 겹치는 픽셀의 수
    inter_px = np.sum(inter_map == 2)

    coverage = inter_px / np.sum(gt_map)

    return coverage


def compare_GT_by_at(raw_img, attr_img, GT_map, topK=100, min_att=0.3):
    # attribution의 map 도출
    attr_boxes = find_topK_rect(raw_img, attr_img, topK, min_att, mode=False)
    attr_bx_maps = get_location_map(attr_boxes, attr_img.shape)
    attr_map = np.clip(sum(attr_bx_maps), 0, 1)

    # GT의 map 계산
    gt_map = np.clip(GT_map, 0, 1)

    # 두개 더해 공통부분 추출
    inter_map = attr_map + gt_map
    # 공통 이미지 중 겹치는 픽셀의 수
    inter_px = np.sum(inter_map == 2)

    coverage = inter_px / np.sum(attr_map)

    return coverage




def Test_perf_byGT(img, GT_map, models, topK=100, min_att=0.3, save='~/python/untitled.xlsx'):
    result_table = df(columns=['No.', 'IG coverage', 'sumIG coverage'])

    for i in range(len(img)):
        # for i in range(100,200):
        t_img = img[i]

        # IG
        t_IG = integrated_gradients(t_img, models)
        if np.sum(t_IG) == 0:
            continue
        v_IG = compare_GT_by_gt(t_img, t_IG, GT_map[i], topK, min_att)

        # sumIG
        t_sumIG = sum_integrated_gradient(t_img, models)
        if np.sum(t_sumIG) == 0:
            continue
        t_sumIG = convert_4Dto3D(sum_Ushape_list(t_sumIG))
        v_sumIG = compare_GT_by_gt(t_img, t_sumIG, GT_map[i], topK, min_att)

        result_table.loc[i] = [str(i), v_IG, v_sumIG]
        global result_table_bygt
        result_table_bygt.loc[i] = [str(i), v_IG, v_sumIG]
        print("Done: {} of {}".format(i + 1, len(img)))

    writer = pd.ExcelWriter(save)
    result_table.to_excel(writer, 'Sheet1')
    writer.save()
    print("Result save in : {}".format(save))
    return result_table


def Test_perf_byAT(img, GT_map, models, topK=100, min_att=0.3, save='untitled.xlsx'):
    result_table = df(columns=['No.', 'IG coverage', 'sumIG coverage'])

    for i in range(len(img)):
        # for i in range(100,200):
        t_img = img[i]

        # IG
        t_IG = integrated_gradients(t_img, models)
        if np.sum(t_IG) == 0:
            continue
        v_IG = compare_GT_by_at(t_img, t_IG, GT_map[i], topK, min_att)

        # sumIG
        t_sumIG = sum_integrated_gradient(t_img, models)
        if np.sum(t_sumIG) == 0:
            continue
        t_sumIG = convert_4Dto3D(sum_Ushape_list(t_sumIG))
        v_sumIG = compare_GT_by_at(t_img, t_sumIG, GT_map[i], topK, min_att)

        result_table.loc[i] = [str(i), v_IG, v_sumIG]
        global result_table_byat
        result_table_byat.loc[i] = [str(i), v_IG, v_sumIG]
        print("Done: {} of {}".format(i + 1, len(img)))

    writer = pd.ExcelWriter(save)
    result_table.to_excel(writer, 'Sheet1')
    writer.save()
    print("Result save in : {}".format(save))
    return result_table