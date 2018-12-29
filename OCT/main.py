import os

import pickle
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from Utils.ExpressiveGradients import *
from Vis.VisualizationTool import *

# Check available GPUs.
print("Available GPUs: {}".format(torch.cuda.device_count()))


class Net(nn.Module):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(14 * 41 * 64, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(200, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(20, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(-1, 14 * 41 * 64)
        out = self.classifier(out)
        # out = F.softmax(out)
        if self.training:
            pass
        else:
            out = F.softmax(out)
        return out

mypath = os.path.dirname(__file__)
model = torch.load(mypath+'/TorchModels/model_NOTnorm_0531.pt')


print(model)


# load images#
f1 = open(mypath+'/Data/x_train_total', 'rb')
x_train_total = pickle.load(f1)
f1.close()

f2 = open(mypath+'/Data/y_train_total', 'rb')
y_train_total = pickle.load(f2)
f2.close()

f3 = open(mypath+'/Data/x_label_total', 'rb')
x_label_total = pickle.load(f3)
f3.close()

img0 = x_train_total[994]

sumIG_0 = sum_integrated_gradient(img0, model)
sumIG_0 = convert_4Dto3D(sum_norm_list(sumIG_0))

plt.interactive(False)
plt.imshow(find_pos_rect(img0, sumIG_0))
plt.show(block = True)

