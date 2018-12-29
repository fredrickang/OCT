from tqdm import tqdm
from Utils.IG_pytorch import *

class CNNfeatures(nn.Module):
    def __init__(self, selects, models):
        super(CNNfeatures, self).__init__()
        self.select = selects
        self.models = models.module.features

    def forward(self, x):
        results = []
        for name, layer in list(self.models.named_children()):
            x = layer(x)
            if name in self.select:
                results.append(x)
        return results


class CNNPartialNet(nn.Module):
    def __init__(self, selects_int, model):
        super(CNNPartialNet, self).__init__()
        self.features = nn.Sequential(
            *list(model.module.features.children())[selects_int + 1:]
        )
        self.classifier = nn.Sequential(
            *list(model.module.classifier.children())
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(-1, 14 * 41 * 64)
        out = self.classifier(out)
        if self.training:
            pass
        else:
            out = F.softmax(out)
        return out

def feature_integrated_gradients(single_img, model, steps=50):
    '''
    return feature integrated gradient map of given single image
    The provided image must of 3D channel-last image
    The return value of image will be list of 4D channel-first image
    '''
    inter_IG = []
    # natural full-inference to calculate loss
    grads = {}
    img_original = single_img2torch(single_img).cpu()
    _, index, _ = label_and_index_and_value(single_img, model)

    scaled_img = np.asarray([float(i) / steps * single_img for i in range(1, steps + 1)])
    img_var = Variable(reshape2torch(scaled_img), requires_grad=True).float()

    model.cuda().eval()
    output = model(img_var)

    one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().cuda()
    one_hot_output[0][index] = 1

    one_hot = torch.sum(Variable(one_hot_output, requires_grad=True) * output)

    # intermediate inference
    # making intermediate images
    selects = ['1', '4', '6', '9', '11', '14']
    extractor = CNNfeatures(selects, model)
    extractor = extractor.eval().cpu()
    intermediate_imgs = extractor(img_original)

    for k in tqdm(range(len(intermediate_imgs))):
        intermediate_img_np = intermediate_imgs[k].data.numpy()[0]
        inter_scaled_img = np.asarray([float(i) / steps * intermediate_img_np for i in range(1, steps + 1)])
        inter_scaled_var = Variable(torch.from_numpy(inter_scaled_img), requires_grad=True).float()

        partialNet = CNNPartialNet(int(selects[k]), model)
        partialNet = partialNet.eval().cpu()
        results = partialNet(inter_scaled_var)

        # inter_hook = inter_scaled_var.register_hook(extract)
        # partialNet.backward(gradients = one_hot)
        loss = torch.sum(Variable(one_hot_output, requires_grad=True).cpu() * results)
        loss.backward()
        # inter_hook.remove()

        inter_grad_mean = torch.mean(inter_scaled_var.grad, 0, True)

        inter_IG_sub = (inter_grad_mean * intermediate_imgs[k]).data.numpy()

        inter_IG.append(inter_IG_sub)

    return inter_IG


def unpool(img_np):
    '''input dimension must of shape (?, channel, image*image) numpy array
       output will be (?, channel, image*image) numpy array'''
    img = img_np.repeat(2, axis = 2).repeat(2, axis = 3)
    return img

def unpool_and_pad(img_np):
    '''input dimension must of shape (?, channel, image*image) numpy array
       output will be same padding (1), with (?, channel, (image+2)*(image+2)) numpy array'''
    img_np = img_np.repeat(2, axis = 2).repeat(2, axis = 3)
    img_tsr = torch.from_numpy(img_np)
    p2d = (1,1,1,1)
    img_var = F.pad(img_tsr.float(), p2d, mode = "replicate")
    img = img_var.data.numpy()
    return img


def im2col(img_cf, h_img, w_img):
    '''input must shape of (1, channel, h, w), 4D image
       output will (h*w, channel) 2D matrix'''
    col_img = np.zeros((h_img * w_img, 1))
    for i in range(img_cf[0].shape[0]):
        col_img = np.append(col_img, img_cf[0][i].reshape((h_img * w_img, 1)), axis=1)

    return np.delete(col_img, 0, axis=1)


def col2im(col_img, h_img, w_img, ch_num):
    '''input must of shape (h*w, channel) 2D matrix
       output will be (channel, h_img, w_img) 3D array'''
    recov_img = col_img.reshape((h_img, w_img, ch_num))
    recov_img = np.transpose(recov_img, (2, 0, 1))
    return recov_img

def parameter_grad_dict(model):
    grad_of_param = {}
    for _, parameter in model.module.features.named_parameters():
        grad_of_param[str(list(parameter.grad.size())[0:2])] = parameter.grad
    return grad_of_param


def reduction(img_np, target_ch, grad_of_param):
    '''input must of shape (1, channel, h, w) 4D numpy image
       output will be (1, target_ch, h, w)
       it require dictionary of gradient of parameters.
       use "parameter_grad_dict" funtion to get it.

       TODO: np.matamul to torch operation?
       '''
    ch_list = [img_np.shape[1], target_ch]
    tmp_grad_var = grad_of_param[str(ch_list)]
    tmp_grad_var = torch.sum(tmp_grad_var, 3)
    tmp_grad_var = torch.sum(tmp_grad_var, 2)
    tmp_grad_var = tmp_grad_var.cpu()

    imcol = im2col(img_np, img_np.shape[2], img_np.shape[3])
    reduce = np.matmul(imcol, tmp_grad_var.data.numpy())
    colim = col2im(reduce, img_np.shape[2], img_np.shape[3], target_ch)
    return np.expand_dims(colim, axis=0)


def sum_integrated_gradient(single_img, models, steps=50):
    pure_IG = integrated_gradients(single_img, models, steps)  # to make model gradient, run FIRST

    # Get parameter gradient and feature integrated gradient images
    grad_dict = parameter_grad_dict(models)
    feat_grad_img = feature_integrated_gradients(single_img, models)
    feat_grad_img = feat_grad_img[::-1]
    # Append raw image
    # pure_IG = integrated_gradients(single_img, model, steps)
    feat_grad_img.append(np.expand_dims(np.transpose(single_img, (2, 0, 1)), axis=0))

    # accumulate images
    accum_img = []
    # accum_img.append(np.expand_dims(np.transpose(pure_IG, (2,0,1)), axis = 0))
    for i in range(len(feat_grad_img) - 1):
        tmp = feat_grad_img[i]
        j = i
        while (tmp.shape[1] != 3):
            if tmp.shape[1] == feat_grad_img[j + 1].shape[1]:
                tmp = unpool_and_pad(tmp)
                # tmp = unpool_and_pad(tmp)[0]
            else:
                tmp = reduction(tmp, feat_grad_img[j + 1].shape[1], grad_dict)
                # tmp = reduction(tmp, feat_grad_img[j+1].shape[1])[0]
            j = j + 1

        if tmp.shape[2] != pure_IG.shape[0]:
            tmp_tsr = torch.from_numpy(tmp)
            pad = (0, 0, 1, 1)
            tmp_var = F.pad(tmp_tsr.float(), pad, mode="replicate")
            tmp = tmp_var.data.numpy()
        elif tmp.shape[3] != pure_IG.shape[1]:
            tmp.tsr = torch.from_numpy(tmp)
            pad = (1, 1)
            tmp_var = F.pad(tmp_tsr.float(), pad, mode="replicate")
            tmp = tmp_var.data.numpy()

        accum_img.append(tmp)
        # accum_img.append(np.expand_dims(tmp, axis = 0))
        accum_img.append(np.expand_dims(np.transpose(pure_IG, (2, 0, 1)), axis=0))

    return accum_img
    # return np.asarray(accum_img)

