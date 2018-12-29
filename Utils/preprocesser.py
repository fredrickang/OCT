import os
import re

import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import multiprocessing
from multiprocessing import cpu_count
import matplotlib.pyplot as plt


#### OCT image path is required #####
path_to_img = '/home/user/'
#path_to_img = 'home/lims1/python/data/OCT_image/'
#####################################


def LoadOCT(Type, Eye, Directory, NumImage=25):
    '''Type: Normal, Dry, Wet
       Eye: OD1 or OD2
       Dir = saved path of OCT images (~/"type" OCT/)
       NumImage = number of OCT images per patients

       Return will be [Image, ImagePath, Type]'''

    # 폴더 내 환자번호를 리스트로 저장
    Dir = Directory
    NameFolder = os.listdir(Dir)
    if '.DS_Store' in NameFolder: # for MacOSX
        NameFolder.remove('.DS_Store')

    if Type == 'Wet':
        bdx = re.compile('^##')
        if list(filter(bdx.match, NameFolder))[0] in NameFolder:
            NameFolder.remove(list(filter(bdx.match, NameFolder))[0])

    print('Number of Patients in ' + Type + ' ' + Eye + ' : {}'.format(len(NameFolder)))

    # Wet의 경우 class가 두가지로 나누어짐
    if Type == 'Wet':
        label_df_wet = pd.read_excel(path_to_img + "/wet AMD 20170828.xlsx")
        label_df_wet.rename(columns={'OCT 판독1': 'OCT opinion1', 'OCT판독2': 'OCT opinion2', 'OCT판독3': 'OCT opinion3'},
                            inplace=True)
        y_label_wet = label_df_wet.loc[:, ['ChartNo.', 'Right/Left eye', 'OCT date', 'plan']]
        y_label_wet['ChartNo.'] = y_label_wet['ChartNo.'].astype(str)

    # 해당 Eye (OD1 or OD2)가 존재하는 폴더만 뽑아옴
    NameFolder_eye = []
    EyeRule = re.compile('^' + Eye)
    for f in tqdm(NameFolder):
        dir_list = os.listdir(Dir + '{}/'.format(f))
        filt = list(filter(EyeRule.match, dir_list))
        if len(filt) == 0:
            NameFolder_eye.append('0')
        else:
            NameFolder_eye.append(filt)

    # 리스트를 벡터로 변환
    NameFolder_eye = [x for y in NameFolder_eye for x in y]

    # 탐색할 디렉토리 주소 지정
    directory = [None] * len(NameFolder)
    for x in range(len(NameFolder)):
        directory[x] = NameFolder[x] + '/' + NameFolder_eye[x]

    # 환자 당 이미지 지정
    NumImage = NumImage

    # 환자별 이미지 번호 XXX형태 통일
    seq = [str(x) for x in range(NumImage)]
    for x in range(len(seq)):
        if len(seq[x]) == 1:
            seq[x] = '00' + seq[x]
        elif len(seq[x]) == 2:
            seq[x] = '0' + seq[x]

    # 이미지 불러오기
    x_label = []
    x_train = []
    y_train = []

    if Type != 'Wet':
        for n in tqdm(directory):
            full_dir = Dir + '{}/'.format(n)
            if not os.path.exists(full_dir):
                continue
            else:
                name = os.listdir(full_dir)[1][:-7]

                # img_dir_list = [(full_dir+name+'{}.jpg'.format(f)) for f in seq]
                # process_img = multiprocessing.Pool(cpu_count()).map(cv2.imread, img_dir_list)
                # final_img = [img[0] for img in process_img]
                # label_list = [(n+'/'+f) for f in seq]

                for f in seq:
                    img = cv2.imread(full_dir + name + '{}.jpg'.format(f))
                    x_train.append(img)
                    x_label.append(n + '/' + name + f + '.jpg')
                    y_train.append(Type)


    elif Type == 'Wet':

        if Eye[0:2] == 'OD':
            eye = 1
        elif Eye[0:2] == 'OS':
            eye = 2

        for n in tqdm(directory):
            full_dir = Dir + '{}/'.format(n)
            if not os.path.exists(full_dir):
                continue
            else:
                name = os.listdir(full_dir)[8][:-7]
                index = np.array(y_label_wet['ChartNo.'] == n.split('/')[0]) & np.array(
                    y_label_wet['Right/Left eye'] == eye)
                plan = list(y_label_wet.loc[index]['plan'])[0]
                for f in seq:
                    img = cv2.imread(full_dir + name + '{}.jpg'.format(f))
                    x_train.append(img)
                    x_label.append(n + '/' + name + f + '.jpg')

                    if plan == 0:
                        y_train.append('observation-Wet')
                    else:
                        y_train.append('anti-VEGF')

    print('Total Image for ' + Type + ' ' + Eye + ' : {}'.format(len(x_train)))

    return x_train, x_label, y_train


def cropOCT(img):
    '''cropping OCT image.
    Take input with array of multiple image with BGR channel.
    Convert image to gray channel, and using binary thresholds
    Use openCV python'''

    img_array = []

    for i in tqdm(range(len(img))):
        threshold_horiz = 0.3 * img[i].shape[1] * 255
        threshold_vert = 0.7 * img[i].shape[0] * 255

        img_gray = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY)

        horiz_sum = np.sum(thresh, axis=1)
        for j in range(50, horiz_sum.shape[0]):
            if horiz_sum[j] < threshold_horiz:
                target_point_x = j
                break

        vert_sum = np.sum(thresh, axis=0)
        for k in range(50, vert_sum.shape[0]):
            if vert_sum[k] < threshold_vert:
                target_point_y = k
                break

        img_crop = img[i][:target_point_x, target_point_y:]

        img_array.append(img_crop)

    return (img_array)


def cropOCT_core(img):
    threshold_horiz = 0.3 * img.shape[1] * 255
    threshold_vert = 0.7 * img.shape[0] * 255

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY)

    horiz_sum = np.sum(thresh, axis=1)
    for j in range(50, horiz_sum.shape[0]):
        if horiz_sum[j] < threshold_horiz:
            target_point_x = j
            break

    vert_sum = np.sum(thresh, axis=0)
    for k in range(50, vert_sum.shape[0]):
        if vert_sum[k] < threshold_vert:
            target_point_y = k
            break

    img_crop = img[:target_point_x, target_point_y:]

    return img_crop


def cropOCT_parallel(img):
    pool = multiprocessing.Pool(cpu_count())
    img_array = []
    for img_crop in pool.map(cropOCT_core, [img[i] for i in range(len(img))]):
        img_array.append(img_crop)
    return (img_array)



if __name__ == '__main__':

    ########################  For Normal OCT Image #######################

    x_train_normal_OD1, x_label_normal_OD1, y_train_normal_OD1 = LoadOCT(
        'Normal', 'OD1', path_to_img + '/Normal OCT/', 25)
    x_train_normal_OD2, x_label_normal_OD2, y_train_normal_OD2 = LoadOCT(
        'Normal', 'OD2', path_to_img + '/Normal OCT/', 25)
    x_train_normal_OS1, x_label_normal_OS1, y_train_normal_OS1 = LoadOCT(
        'Normal', 'OS1', path_to_img + '/Normal OCT/', 25)
    x_train_normal_OS2, x_label_normal_OS2, y_train_normal_OS2 = LoadOCT(
        'Normal', 'OS2', path_to_img + '/Normal OCT/', 25)

    # Handling exception
    x_train_normal_OS2[424] = x_train_normal_OS2[423]

    x_train_normal_horiz = x_train_normal_OD1 + x_train_normal_OS1
    x_label_normal_horiz = x_label_normal_OD1 + x_label_normal_OS1
    y_train_normal_horiz = y_train_normal_OD1 + y_train_normal_OS1

    x_train_normal_vert = x_train_normal_OD2 + x_train_normal_OS2
    x_label_normal_vert = x_label_normal_OD2 + x_label_normal_OS2
    y_train_normal_vert = y_train_normal_OD2 + y_train_normal_OS2

    ########################  For Dry OCT Image #######################

    x_train_dry_OD1, x_label_dry_OD1, y_train_dry_OD1 = LoadOCT(
        'observation-Dry', 'OD1', path_to_img + '/Dry AMD/')
    x_train_dry_OD2, x_label_dry_OD2, y_train_dry_OD2 = LoadOCT(
        'observation-Dry', 'OD2', path_to_img + '/Dry AMD/')
    x_train_dry_OS1, x_label_dry_OS1, y_train_dry_OS1 = LoadOCT(
        'observation-Dry', 'OS1', path_to_img + '/Dry AMD/')
    x_train_dry_OS2, x_label_dry_OS2, y_train_dry_OS2 = LoadOCT(
        'observation-Dry', 'OS2', path_to_img + '/Dry AMD/')

    #Handling exception
    x_train_dry_OD2[49] = x_train_dry_OD2[48]
    x_train_dry_OD2[424] = x_train_dry_OD2[423]

    x_train_dry_horiz = x_train_dry_OD1 + x_train_dry_OS1
    x_label_dry_horiz = x_label_dry_OD1 + x_label_dry_OS1
    y_train_dry_horiz = y_train_dry_OD1 + y_train_dry_OS1

    x_train_dry_vert = x_train_dry_OD2 + x_train_dry_OS2
    x_label_dry_vert = x_label_dry_OD2 + x_label_dry_OS2
    y_train_dry_vert = y_train_dry_OD2 + y_train_dry_OS2

    indexes = [i for i, x in enumerate(x_train_dry_vert) if x == None]
    x_label_dry_vert = [i for j, i in enumerate(x_label_dry_vert) if j not in indexes]
    y_train_dry_vert = [i for j, i in enumerate(y_train_dry_vert) if j not in indexes]
    x_train_dry_vert = [x for x in x_train_dry_vert if x is not None]
    for i in range(len(x_train_dry_vert)):
        if x_train_dry_vert[i] == None:
            print(i)


    ########################  For Wet OCT Image #######################

    x_train_wet_OD1, x_label_wet_OD1, y_train_wet_OD1 = LoadOCT(
        'Wet', 'OD1', path_to_img + '/Wet AMD/')
    x_train_wet_OD2, x_label_wet_OD2, y_train_wet_OD2 = LoadOCT(
        'Wet', 'OD2', path_to_img + '/Wet AMD/')
    x_train_wet_OS1, x_label_wet_OS1, y_train_wet_OS1 = LoadOCT(
        'Wet', 'OS1', path_to_img + '/Wet AMD/')
    x_train_wet_OS2, x_label_wet_OS2, y_train_wet_OS2 = LoadOCT(
        'Wet', 'OS2', path_to_img + '/Wet AMD/')

    #Handling exception
    x_train_wet_OD2[49] = x_train_wet_OD2[48]

    x_train_wet_horiz = x_train_wet_OD1 + x_train_wet_OS1
    x_label_wet_horiz = x_label_wet_OD1 + x_label_wet_OS1
    y_train_wet_horiz = y_train_wet_OD1 + y_train_wet_OS1

    x_train_wet_vert = x_train_wet_OD2 + x_train_wet_OS2
    x_label_wet_vert = x_label_wet_OD2 + x_label_wet_OS2
    y_train_wet_vert = y_train_wet_OD2 + y_train_wet_OS2


    ################# Cropping ###################################
    x_train_normal_horiz = cropOCT(x_train_normal_horiz)
    x_train_normal_vert = cropOCT(x_train_normal_vert)
    x_train_dry_horiz = cropOCT(x_train_dry_horiz)
    x_train_dry_vert = cropOCT(x_train_dry_vert)
    x_train_wet_horiz = cropOCT(x_train_wet_horiz)
    x_train_wet_vert = cropOCT(x_train_wet_vert)


    ################# Image Resizing ############################

    for x in tqdm(range(len(x_train_normal_horiz))):
        im = x_train_normal_horiz[x]
        imS = cv2.resize(im, (342, 128))
        block = imS[110:, 320:]
        imS[110:, 0:22] = block
        x_train_normal_horiz[x] = imS

    for x in tqdm(range(len(x_train_normal_vert))):
        im = x_train_normal_vert[x]
        imS = cv2.resize(im, (342, 128))
        block = imS[110:, 320:]
        imS[110:, 0:22] = block
        x_train_normal_vert[x] = imS

    for x in tqdm(range(len(x_train_dry_horiz))):
        im = x_train_dry_horiz[x]
        imS = cv2.resize(im, (342, 128))
        block = imS[110:, 320:]
        imS[110:, 0:22] = block
        x_train_dry_horiz[x] = imS

    for x in tqdm(range(len(x_train_dry_vert))):
        im = x_train_dry_vert[x]
        imS = cv2.resize(im, (342, 128))
        block = imS[110:, 320:]
        imS[110:, 0:22] = block
        x_train_dry_vert[x] = imS

    for x in tqdm(range(len(x_train_wet_horiz))):
        im = x_train_wet_horiz[x]
        imS = cv2.resize(im, (342, 128))
        block = imS[110:, 320:]
        imS[110:, 0:22] = block
        x_train_wet_horiz[x] = imS

    for x in tqdm(range(len(x_train_wet_vert))):
        im = x_train_wet_vert[x]
        imS = cv2.resize(im, (342, 128))
        block = imS[110:, 320:]
        imS[110:, 0:22] = block
        x_train_wet_vert[x] = imS


    ############### Check data #####################
    print('############ Normal Data ############')
    print("\n")
    print(len(x_train_normal_horiz))
    print(len(y_train_normal_horiz))
    print(len(x_label_normal_horiz))
    print(len(x_train_normal_vert))
    print(len(y_train_normal_vert))
    print(len(x_label_normal_vert))
    print("\n")
    print("\n")

    print('############ Dry Data ############')
    print("\n")
    print(len(x_train_dry_horiz))
    print(len(y_train_dry_horiz))
    print(len(x_label_dry_horiz))
    print(len(x_train_dry_vert))
    print(len(y_train_dry_vert))
    print(len(x_label_dry_vert))
    print("\n")
    print("\n")

    print('############ Dry Data ############')
    print("\n")
    print(len(x_train_wet_horiz))
    print(len(y_train_wet_horiz))
    print(len(x_label_wet_horiz))
    print(len(x_train_wet_vert))
    print(len(y_train_wet_vert))
    print(len(x_label_wet_vert))
    print("\n")
    print("\n")

    print('###### Rectangled Image size ####')
    print("\n")
    for i in range(100, 105):
        plt.figure(i + 1)
        plt.imshow(x_train_wet_horiz[i])
        print(x_train_wet_horiz[i].shape)

    print('##############################')
    print('###### preprocessing Done ####')
    print('##############################')



