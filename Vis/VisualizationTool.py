#=====================================================#
#           Visualization tool for OCTXAI             #
#=====================================================#

from Utils.utility import *
from Vis import *
from tqdm import tqdm
import copy
import cv2
import imutils
from imutils import contours
import itertools


def plot1Dimg(img):
    '''input: image with (height * width) / no channel'''
    img_np = np.transpose([img, img, img], axes = [1,2,0])
    plt.imshow(img_np)

def plot_batch_img(single_img_4D):
    img = np.transpose(single_img_4D, (0,2,3,1))
    plt.imshow(img[0])

def show_img(single_img, attrs, model):
    label, index, value = label_and_index_and_value(single_img, model)
    print("Label: %s   /   Score: %s" % (label, value))

    plt.figure(1)
    plt.imshow(single_img)
    plt.figure(2)
    plt.imshow(attrs)
    # plt.imshow(np.uint8(attrs))

def visualizing_attrs_windowing(img, attrs, ptile = 99):
    attrs = gray_scale(attrs)
    attrs = abs(attrs)
    attrs = np.clip(attrs/np.percentile(attrs, ptile), 0, 1)
    vis = img * attrs
    plt.imshow(np.uint8(vis))

def visualizing_attrs_overlay(img, attrs, pos_ch = G, neg_ch = R, ptile = 99):
    attrs = gray_scale(attrs)
    #attrs = att_ch_sum(attrs)
    attrs = normalize(attrs, ptile)
    pos_attrs = attrs * (attrs >= 0.0)
    neg_attrs = attrs * (attrs < 0.0)
    attrs_mask = pos_attrs*pos_ch + neg_attrs*neg_ch
    vis = 0.3*gray_scale(img) + 0.7*attrs_mask
    #vis = 0.1*gray_scale(img) + 0.9*attrs_mask
    plt.imshow(np.uint8(vis))

def visualizing_positive_attrs_overlay(img, attrs, pos_ch = R, ptile = 99):
    attrs = gray_scale(attrs)
    #attrs = att_ch_sum(attrs)
    attrs = normalize(attrs, ptile)
    pos_attrs = attrs * (attrs >= 0.0)
    attrs_mask = pos_attrs*pos_ch
    vis = 0.3*gray_scale(img) + 0.7*attrs_mask
    #vis = 0.1*gray_scale(img) + 0.9*attrs_mask
    plt.imshow(np.uint8(vis))

def plot_all(raw_img, att_img, model, topK = 100, min_att=0.3, ptile = 99):
    R = np.array([255,0,0])
    plt.figure(1)
    plt.figure(2)
    show_img(raw_img, att_img, model)
    plt.figure(3)
    visualizing_positive_attrs_overlay(raw_img, att_img, R, ptile)
    plt.figure(4)
    plt.imshow(find_topK_rect(raw_img, att_img, topK, min_att))


def kernel_plot_1(channel_img_4D):
    channel_img = channel_img_4D[0]
    for i in range(len(channel_img[0])):
        print("kernel No. : " + str(i + 1))
        plt.figure(i + 1)
        plt.imshow(np.transpose(np.array([channel_img[i], channel_img[i], channel_img[i]]), axes=[1, 2, 0]))


def kernel_plot(channel_img_4D):
    channel_img = channel_img_4D[0]
    col = 4
    row = int(channel_img.shape[0] / 4)
    #fig, ax = plt.subplot(col, col)
    plt.figure(figsize=(10,10))
    for i in tqdm(range(len(channel_img))):
        ax = plt.subplot(row, col, i+1)
        ax.imshow(np.transpose(np.array([channel_img[i], channel_img[i], channel_img[i]]), axes = [1,2,0]))


def find_pos_pixel(raw_img, attr_img, min_att=0.3, round_over=2, circle_size=5, circle_type=-1):
    '''min_att: cut-off value for integrated gradient (lower bound)
       round_over: round option
       circle_size: how many neighborhood-pixels will be considered to visualize (circle size)
       cirvle_type: the circle type (-1 will gill out the whole circle, otherwise it means width of the line)
    '''
    attr_img = gray_scale(attr_img)
    attr_img = attr_img * (attr_img >= 0.0)
    attr_img = np.mean(attr_img, axis=2)

    # Mean std norm 제거
    # attr_img = normstd(attr_img)
    attr_img = norm01(attr_img)
    pos_position = np.array(np.nonzero(np.round(attr_img, round_over) > min_att))
    img = copy.deepcopy(raw_img)
    for i in range(pos_position[0].size):
        img = cv2.circle(img, (pos_position[1, i], pos_position[0, i]), circle_size, (255, 0, 0), circle_type)
    return img


def find_pos_rect(raw_img, attr_img, min_att=0.3, round_over=2, circle_size=5, circle_type=-1, mode=True):
    cir_img = find_pos_pixel(raw_img, attr_img, min_att, round_over, circle_size, -1)
    cir_img_red = cir_img[:, :, 0]
    thresh = cv2.threshold(cir_img_red, 254.5, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = contours.sort_contours(cnts)[0]

    boxes = []
    rect_img = copy.deepcopy(raw_img)
    for (i, c) in enumerate(cnts):
        # (x,y,w,h) = cv2.boundingRect(c)
        if cv2.contourArea(c) < 150:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)

        box = np.int0(box)
        boxes.append(box)
        rect_img = cv2.drawContours(rect_img, [box], 0, (255, 0, 0), 2)

    if mode:
        return rect_img
    else:
        return boxes


def get_location_map(box_list, dim):
    def area_check(boxpoints, iteridx, mode=True):
        mat = np.zeros(dim)
        if mode:
            area = dist(boxpoints[0], boxpoints[1]) * dist(boxpoints[3], boxpoints[0])
        else:
            area = dist(boxpoints[0], boxpoints[1]) * dist(boxpoints[1], boxpoints[2])

        for idx in iteridx:
            t1 = triangle_area(boxpoints[0][1], boxpoints[0][0], boxpoints[1][1], boxpoints[1][0], idx[0], idx[1])
            t2 = triangle_area(boxpoints[1][1], boxpoints[1][0], boxpoints[2][1], boxpoints[2][0], idx[0], idx[1])
            t3 = triangle_area(boxpoints[2][1], boxpoints[2][0], boxpoints[3][1], boxpoints[3][0], idx[0], idx[1])
            t4 = triangle_area(boxpoints[3][1], boxpoints[3][0], boxpoints[0][1], boxpoints[0][0], idx[0], idx[1])
            t0 = t1 + t2 + t3 + t4
            if (t0 * 0.9) <= area:
                mat[idx[0], idx[1], :] = 1

        return mat

    mat_list = []
    for bx in tqdm(box_list):
        idx_iter = list(itertools.product(range(dim[0]), range(dim[1])))
        mat_map = area_check(bx, idx_iter, mode=True)
        if np.sum(mat_map) == 0:
            mat_map = area_check(bx, idx_iter, mode=False)
        mat_list.append(mat_map)
    return mat_list


def find_topK_rect(raw_img, attr_img, topK, min_att=0.3, round_over=2, circle_size=5, circle_type=-1, mode=True):
    boxes = find_pos_rect(raw_img, attr_img, min_att, round_over, circle_size, -1, mode=False)
    bx_map = get_location_map(boxes, raw_img.shape)
    pos_att = attr_img * (attr_img >= 0)
    bx_imp = []
    for i in range(len(bx_map)):
        bx_val = pos_att * bx_map[i]
        bx_imp.append(np.sum(bx_val) / np.count_nonzero(bx_val))

    argsort = np.argsort(np.array(bx_imp))
    boxes = [boxes[i] for i in argsort[::-1]]
    if not mode:
        return boxes[:min(topK, len(bx_map))]
    else:
        rect_img = copy.deepcopy(raw_img)
        for k in range(min(topK, len(bx_map))):
            rect_img = cv2.drawContours(rect_img, [boxes[k]], 0, (255, 0, 0), 2)

        return rect_img

