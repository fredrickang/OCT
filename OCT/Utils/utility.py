from Utils import *

def reshape2torch(batch_img):
    '''
    input : scaled image for Integrated Gradint
    or, batch of images (4D): channel last t0 channel first
    '''
    img_tsr = torch.from_numpy(np.transpose(batch_img, (0,3,1,2)))
    return img_tsr

def single_img2torch(single_img):
    '''
    input: single image
    output: Variable for torch with float type
    '''
    img_np = np.transpose(single_img, (2,0,1))
    img_tsr = torch.from_numpy(img_np).unsqueeze(0)
    img_var = Variable(img_tsr).float()
    return img_var


def label_and_index_and_value(single_img, model):
    '''find the label and score of given single image'''

    # img_tsr = torch.from_numpy(np.transpose(single_img, (2,0,1))).unsqueeze(0)
    # img_var = Variable(img_tsr).float()
    img_var = single_img2torch(single_img)
    model.cuda().eval()
    output = model(img_var)
    output.np = output.data[0].cpu()

    value, index = torch.max(output, 1)
    label = labels[index.data.cpu().numpy()[0]]
    return label, index.data.cpu().numpy()[0], value.data.cpu().numpy()[0]

def gray_scale(img):
    '''Converts the provided RGB image to gray scale
       The provided input must of shape 3D channel-last, with (R,G,B)'''
    img = np.average(img, axis = 2)
    return np.transpose([img,img,img], axes = [1,2,0])

def att_ch_sum(attr):
    attrs = np.sum(attr, axis = 2)
    return np.transpose([attrs, attrs, attrs], axes = [1,2,0])

def normalize(attrs, ptile = 99):
    '''Normalize the provided attribution image [-1.0, 1.0]'''
    h = np.percentile(attrs, ptile)
    l = np.percentile(attrs, 100-ptile)
    return np.clip(attrs/max(abs(h), abs(l)), -1.0, 1.0)

def normalize01(attrs, ptile = 99):
    '''Normalize the provided attribution image [0.0, 1.0]'''
    h = np.percentile(attrs, ptile)
    l = np.percentile(attrs, 100-ptile)
    return np.clip(attrs/max(abs(h), abs(l)), 0.0, 1.0)

def normstd(attrs):
    img = (attrs - np.mean(attrs)) / np.std(attrs)
    return np.clip(img, 0.0, 255.)

def norm01(img):
    return (img - np.min(img))/(np.max(img) - np.min(img))

def convert_4Dto3D(img_4d):
    return np.transpose(img_4d[0], (1,2,0))

def sum_norm_list(list):
    output = -1 + 2*(list[0] - np.min(list[0]))/(np.max(list[0]) - np.min(list[0]))
    for i in range(1,len(list)):
        tmp = -1 + 2*(list[i] - np.min(list[i]))/(np.max(list[i]) - np.min(list[i]))
        output = output + tmp
    return output

def norm11(img):
    return -1 + 2*(img - np.min(img))/(np.max(img) - np.min(img))

def sum_fore_list(list):
    return 0.1*norm11(list[0]) + 0.13*norm11(list[1]) + 0.16*norm11(list[2]) + 0.18*norm11(list[3]) + 0.20*norm11(list[4]) + 0.23*norm11(list[5]) + 2*norm11(list[6])

def sum_aft_list(list):
    return 0.23*norm11(list[0]) + 0.20*norm11(list[1]) + 0.18*norm11(list[2]) + 0.16*norm11(list[3]) + 0.13*norm11(list[4]) + 0.1*norm11(list[5]) + 0.05*norm11(list[6])

def sum_Ushape_list(list):
    return 3*norm11(list[0]) + 0.5*norm11(list[1]) + 0.5*norm11(list[2]) + 0.5*norm11(list[3]) + 0.5*norm11(list[4]) + 0.5*norm11(list[5]) + 3*norm11(list[6])

def triangle_area(x1, y1, x2, y2, x3, y3):
    return abs(0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)))

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))
