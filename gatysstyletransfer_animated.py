import imageio
import os
import sys
import ntpath
from os import listdir
from os.path import join, isfile
from PIL import Image
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# constants
style_gif = sys.argv[1]
content_gif = sys.argv[2]


def load_files(path):
    files = []
    path = path
    for file in os.listdir(path):
        fPath = join(path, file)
        if isfile(fPath):
            img = Image.open(fPath)
            f_name = os.path.splitext(ntpath.basename(fPath))[0]
            files.append(f_name)
    return files


num_key_frames = 15
image_path = "content/"
style_image_path = "style/"
image_names = load_files("content")
style_images = load_files("style")


# vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return (out)


# pre and post processing for images
image_size = 512
prep = transforms.Compose([transforms.Scale(image_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                std=[1, 1, 1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                           ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                             transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                  std=[1, 1, 1]),
                             transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                             ])
postpb = transforms.Compose([transforms.ToPILImage()])


def postp(tensor):  # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    img = postpb(t)
    return img


# Loading weights to the above self defined VGG architecture

vgg = VGG()
vgg.load_state_dict(torch.load('vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()

im = Image.open(content_gif)
for i in range(num_key_frames):
    im.seek(im.n_frames // num_key_frames * i)
    rgb_im = im.convert('RGB')
    rgb_im.save("content/" + str(i) + '.jpeg')
im = Image.open(style_gif)
for i in range(num_key_frames):
    im.seek(im.n_frames // num_key_frames * i)
    rgb_im = im.convert('RGB')
    rgb_im.save("style/" + str(i) + '.jpeg')



for i in range(5, num_key_frames):
    print("start")
    content_image_name = image_names[i]
    style_image_name = style_images[i]
    img_dirs = [style_image_path, image_path]
    img_names = [style_image_name, content_image_name]
    imgs = [Image.open(img_dirs[i] + name + ".jpeg") for i, name in enumerate(img_names)]
    imgs_torch = [prep(img) for img in imgs]
    if torch.cuda.is_available():
        imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
    style_image, content_image = imgs_torch

    opt_img = Variable(content_image.data.clone(), requires_grad=True)

    style_layers = ['r11', 'r21']

    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    loss_functions = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    if torch.cuda.is_available():
        loss_functions = [loss_function.cuda() for loss_function in loss_functions]

    style_weights = [1e3 / n ** 2 for n in [64, 128]]
    content_weights = [1e0]
    weights = style_weights + content_weights

    style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    targets = style_targets + content_targets

    # run style transfer
    max_iter = 500
    show_iter = 50
    optimizer = optim.LBFGS([opt_img]);
    n_iter = [0]

    while n_iter[0] <= max_iter:

        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            layer_losses = [weights[a] * loss_functions[a](A, targets[a]) for a, A in enumerate(out)]
            loss = sum(layer_losses)
            loss.backward()
            n_iter[0] += 1
            if n_iter[0] % show_iter == (show_iter - 1):
                print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
            return loss


        optimizer.step(closure)

    out_img1 = postp(opt_img.data[0].cpu().squeeze())
    out_image_name = 'transfer frames/' + content_image_name + ".jpeg"
    out_img1.save(out_image_name)

images = []
files = []
path = "transfer frames"
for file in os.listdir(path):
    fPath = join(path, file)
    if isfile(fPath):
        images.append(imageio.imread(fPath))

imageio.mimsave('transfer frames/movie.gif', images)
