from Utils.utility import *

def integrated_gradients(single_img, model, steps=50):
    '''return integrated gradient map of given single image
       The provided image must of 3D channel-last image
       The return value of image will be 3D channel-last image
       '''
    grads = {}
    # img_original = Variable(torch.from_numpy(np.transpose(single_img, (2,0,1))).unsqueeze(0)).float()
    img_original = single_img2torch(single_img)
    _, index, _ = label_and_index_and_value(single_img, model)

    scaled_img = np.asarray([float(i) / steps * single_img for i in range(1, steps + 1)])
    img_var = Variable(reshape2torch(scaled_img), requires_grad=True).float()

    model.cuda().eval()
    output = model(img_var)

    one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().cuda()
    one_hot_output[0][index] = 1

    one_hot = torch.sum(Variable(one_hot_output, requires_grad=True) * output)
    hook = img_var.register_hook(extract)
    one_hot.backward(retain_graph=True)
    hook.remove()

    grad_mean = torch.mean(yGrad, 0, True)

    IG_return = (grad_mean * img_original).data.numpy()

    return np.transpose(IG_return[0], (1, 2, 0))

def extract(xVar):
    global yGrad
    yGrad = xVar