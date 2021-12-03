import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

image_path = 'Images/'

# constants
content_image_name = ""
style_image_name = "disney.jpg"


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


"""Loading weights to the above self defined VGG architecture"""

vgg = VGG()
vgg.load_state_dict(torch.load('vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()


# load images, ordered as [style_image, content_image]
image_names = ['MLK.jpg', 'sreac.jpg', 'towerhall.jpg', 'cityhall.jpg', 'Crystal.jpeg', 'Evening.jpeg', 'Buildings.jpg',
               'Trees.jpeg', 'Aditya.jpg', 'Pradeep.jpg']

content_image_name = image_names[0]

img_dirs = [image_path, image_path]
img_names = [style_image_name, content_image_name]
imgs = [Image.open(img_dirs[i] + name) for i, name in enumerate(img_names)]
imgs_torch = [prep(img) for img in imgs]
if torch.cuda.is_available():
    imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
else:
    imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
style_image, content_image = imgs_torch

opt_img = Variable(content_image.data.clone(), requires_grad=True)


# display images
i = 0
fig, axes = plt.subplots(1, 2, figsize=(15, 15))
for img in imgs:
    axes[i].imshow(img)

    if i == 0:
        axes[i].set_title('Disney Logo')

    else:
        axes[i].set_title('MLK Building')
    i = i + 1

"""Now, we do experimentation by choosing different layers for Style and Content.\
Firstly we choose <b>'r11' and 'r21'</b> for style and <b>'r42'</b> for content.
"""

style_layers = ['r11', 'r21']

content_layers = ['r42']
loss_layers = style_layers + content_layers
loss_functions = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
if torch.cuda.is_available():
    loss_functions = [loss_function.cuda() for loss_function in loss_functions]

# these are good weights settings:
style_weights = [1e3 / n ** 2 for n in [64, 128]]
content_weights = [1e0]
weights = style_weights + content_weights

# compute optimization targets
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
        # print loss
        if n_iter[0] % show_iter == (show_iter - 1):
            print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
        #             print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
        return loss


    optimizer.step(closure)

# display result
out_img1 = postp(opt_img.data[0].cpu().squeeze())
plt.imshow(out_img1)
plt.gcf().set_size_inches(10, 10)

out_image_name = 'Output_Images/' + 'diney_' + content_image_name
out_img1.save(out_image_name)

# from PIL import Image
# im = Image.open('/content/style.gif')
# for i in range(im.n_frames):
#   im.seek(i)
#   # im = im.convert('RGB')
#   im.save("/content/Style_Frames/"+str(i)+'.png')
# im = Image.open('/content/content.gif')
# for i in range(im.n_frames):
#   im.seek(i)
#   im.save("/content/Content_Frames/"+str(i)+'.png')
# #load images, ordered as [style_image, content_image]

# def load_files(folder_name):
#     files = []
#     path = join(os.getcwd(), folder_name)
#     for file in listdir(path):
#         fPath = join(path, file)
#         if isfile(fPath):
#             img = Image.open(fPath)
#             arr = (np.array(img))
#             files.append(arr)
#             f_name = os.path.splitext(ntpath.basename(fPath))[0]
#             key_list = list(COLLISION_TYPE.keys())
#             val_list = list(COLLISION_TYPE.values())
#             position = val_list.index(f_name.split(".")[0].split("_", 1)[1])
#             class_label.append(key_list[position])
#     return files, class_label
# image_names = ['Building1.jpeg','Building2.jpeg','Building3.jpeg','Building4.jpeg','Glass_Sphere.jpeg',\
#                'Ocean_pink_sky.jpeg','Tree_Sun.jpeg','Building_river.jpeg','Amol.jpeg','Suhrid.jpeg']

# content_image_name = image_names[2]

# img_dirs = [image_path, image_path]
# img_names = ['vangogh.jpg', content_image_name]
# imgs = [Image.open(img_dirs[i] + name) for i,name in enumerate(img_names)]
# imgs_torch = [prep(img) for img in imgs]
# if torch.cuda.is_available():
#     imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
# else:
#     imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
# style_image, content_image = imgs_torch

# # opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
# opt_img = Variable(content_image.data.clone(), requires_grad=True)

# style_layers = ['r11','r21'] 

# content_layers = ['r42']
# loss_layers = style_layers + content_layers
# loss_functions = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
# if torch.cuda.is_available():
#     loss_functions = [loss_function.cuda() for loss_function in loss_functions]

# #these are good weights settings:
# style_weights = [1e3/n**2 for n in [64,128]]
# content_weights = [1e0]
# weights = style_weights + content_weights

# #compute optimization targets
# style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
# content_targets = [A.detach() for A in vgg(content_image, content_layers)]
# targets = style_targets + content_targets


# #run style transfer
# max_iter = 500
# show_iter = 50
# optimizer = optim.LBFGS([opt_img]);
# n_iter=[0]

# while n_iter[0] <= max_iter:

#     def closure():
#         optimizer.zero_grad()
#         out = vgg(opt_img, loss_layers)
#         layer_losses = [weights[a] * loss_functions[a](A, targets[a]) for a,A in enumerate(out)]
#         loss = sum(layer_losses)
#         loss.backward()
#         n_iter[0]+=1
#         #print loss
#         if n_iter[0]%show_iter == (show_iter-1):
#             print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))
# #             print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
#         return loss

#     optimizer.step(closure)
# out_image_name = '/content/Output_Image/'+'Vangogh_'+content_image_name
# out_img1.save(out_image_name)


image_names = ['MLK.jpg', 'sreac.jpg', 'towerhall.jpg', 'cityhall.jpg', 'Crystal.jpeg', 'Evening.jpeg', 'Buildings.jpg',
               'Trees.jpeg', 'Aditya.jpg', 'Pradeep.jpg']


image_dir = "Output_Images/"

imgs = [Image.open('Images/' + name) for i, name in enumerate(image_names)]

imgs_spongebob = [Image.open(image_dir + "spongebob_" + name) for i, name in enumerate(image_names)]
imgs_monalisa = [Image.open(image_dir + 'monalisa_' + name) for i, name in enumerate(image_names)]
imgs_studio_ghibli = [Image.open(image_dir + 'studio_ghibli_' + name) for i, name in enumerate(image_names)]
imgs_disney = [Image.open(image_dir + 'disney_' + name) for i, name in enumerate(image_names)]

fig, axs = plt.subplots(4, 5, figsize=(18, 11), sharex=True, sharey=True)

for i in range(4):
    j = i
    axs[i, 0].imshow(imgs[j].resize(imgs_monalisa[j].size))
    axs[i, 1].imshow(imgs_monalisa[j])
    axs[i, 2].imshow(imgs_disney[j])
    axs[i, 3].imshow(imgs_studio_ghibli[j])
    axs[i, 4].imshow(imgs_spongebob[j])
    axs[i, 0].set_title('Input image')
    axs[i, 1].set_title('Monalisa')
    axs[i, 2].set_title('Disney')
    axs[i, 3].set_title('Studio Ghibli')
    axs[i, 4].set_title('Spongebob Squarepants')

fig, axs = plt.subplots(4, 5, figsize=(18, 11), sharex=True, sharey=True)

for i in range(4):
    j = i + 4
    axs[i, 0].imshow(imgs[j].resize(imgs_monalisa[j].size))
    axs[i, 1].imshow(imgs_monalisa[j])
    axs[i, 2].imshow(imgs_disney[j])
    axs[i, 3].imshow(imgs_studio_ghibli[j])
    axs[i, 4].imshow(imgs_spongebob[j])
    axs[i, 0].set_title('Input image')
    axs[i, 1].set_title('Monalisa')
    axs[i, 2].set_title('Disney')
    axs[i, 3].set_title('Studio Ghibli')
    axs[i, 4].set_title('Spongebob Squarepants')

fig, axs = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)

for i in range(2):
    j = i + 8
    axs[i, 0].imshow(imgs[j].resize(imgs_monalisa[j].size))
    axs[i, 1].imshow(imgs_monalisa[j])
    axs[i, 2].imshow(imgs_disney[j])
    axs[i, 3].imshow(imgs_studio_ghibli[j])
    axs[i, 4].imshow(imgs_spongebob[j])
    axs[i, 0].set_title('Input image')
    axs[i, 1].set_title('Monalisa')
    axs[i, 2].set_title('Disney')
    axs[i, 3].set_title('Studio Ghibli')
    axs[i, 4].set_title('Spongebob Squarepants')


# display result
out_img2 = postp(opt_img.data[0].cpu().squeeze())
plt.imshow(out_img2)
plt.gcf().set_size_inches(10, 10)

"""Let's now use 5 layers r11, r21, r31, r41 and r51 for extracting the style and r42, r32 and r22 for extracting the content."""

style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']

content_layers = ['r42', 'r32', 'r22']
loss_layers = style_layers + content_layers
loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
if torch.cuda.is_available():
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

# these are good weights settings:
style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
content_weights = [1e0, 1e0, 1e0]
weights = style_weights + content_weights

# compute optimization targets
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
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        n_iter[0] += 1
        # print loss
        if n_iter[0] % show_iter == (show_iter - 1):
            print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
        return loss


    optimizer.step(closure)

out_img3 = postp(opt_img.data[0].cpu().squeeze())

plt.imshow(out_img3)
plt.gcf().set_size_inches(10, 10)

"""Lets compare the 3 results we obtained"""

fig, ax = plt.subplots(2, 2, figsize=(15, 10))

ax[0, 0].imshow(imgs[0])
ax[0, 0].title.set_text('Image')

ax[0, 1].imshow(out_img1)
ax[0, 1].title.set_text('First Output')

ax[1, 0].imshow(out_img2)
ax[1, 0].title.set_text('Second Output')

ax[1, 1].imshow(out_img3)
ax[1, 1].title.set_text('Third Output')
