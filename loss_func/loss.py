import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1',
        '31': 'conv5_2'
    }

    features = {}
    x = image

    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def vgg_normalize(image):
    device = image.device

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std

    return image


# to change
class HarmonizationLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        vgg = torchvision.models.vgg19(pretrained=True).features.to(device)
        for x in vgg.parameters():
            x.requires_grad = False
        self.vgg = vgg

        self.cos = torch.nn.CosineSimilarity()


    def forward(self, res, tar):
        res_f = get_features(vgg_normalize(res), self.vgg)
        tar_f = get_features(vgg_normalize(tar), self.vgg)

        loss1 = torch.mean((res_f['conv5_2'] - tar_f['conv5_2']) ** 2, dim=(1, 2, 3))
        loss2 = torch.mean((res_f['conv4_2'] - tar_f['conv4_2']) ** 2, dim=(1, 2, 3))

        return loss1+loss2


class AdversarialLoss(torch.nn.Module):
    def __init__(self, device, model):
        super().__init__()

        self.device = device

        self.loss = nn.CrossEntropyLoss()

        for x in model.parameters():
            x.requires_grad = False
        self.model = model

    def forward(self, ori, res):

        # model输出 batch * type (no softmax)

        target_predict = self.model(ori)
        target_label = torch.argmax(target_predict, dim=1)
        result_predict = self.model(res)
        loss = self.loss(-result_predict, target_label)

        return loss


class Loss(torch.nn.Module):
    def __init__(self,
                 device="cuda",
                 model=nn.Module(),
                 l_h=1,
                 l_a=1,
                 patch_size=128,
                 patch_num=16):

        super().__init__()

        self.device = device

        self.harmonization_loss = HarmonizationLoss(device)
        # self.adversarial_loss = AdversarialLoss(device, model)
        self.l_h = l_h
        self.l_a = l_a

        self.patch_size = patch_size
        self.patch_num = patch_num


    def forward_harmonization(self, res, tar):
        return self.harmonization_loss(res, tar)

    def forward_adversarial(self, ori, res):
        return self.adversarial_loss(ori, res)

    def random_patch_points(self, img_shape, num_patches, size):
        batch_size, channels, height, width = img_shape
        half = size // 2

        w = torch.randint(low=half, high=width - half, size=(num_patches, 1))
        h = torch.randint(low=half, high=height - half, size=(num_patches, 1))
        points = torch.cat([h, w], dim=1)

        points = points.to(self.device)

        return points

    def generate_patches(self, img: torch.Tensor, patch_points, size):

        batch_size, channels, height, width = img.shape

        num_patches = patch_points.shape[0]

        patches = []
        half = size // 2

        for patch_idx in range(num_patches):
            point_x = patch_points[patch_idx][0]
            point_y = patch_points[patch_idx][1]
            patch = img[:, :, point_y - half:point_y + half, point_x - half:point_x + half]
            patch = F.interpolate(patch, (height, width), mode="bilinear", align_corners=True)
            patches.append(patch)

        patches = torch.cat(patches, dim=0)

        return patches

    def forward_patch(self, res, tar):

        points = self.random_patch_points(res.shape, self.patch_num, self.patch_size)

        res_patches = self.generate_patches(res, points, self.patch_size)
        tar_patches = self.generate_patches(tar, points, self.patch_size)

        loss_ori = self.harmonization_loss(res_patches, tar_patches)

        with torch.no_grad():
            loss_avg = self.harmonization_loss(res, tar)
            loss_avg = loss_avg.unsqueeze(1)

            shape = list(loss_avg.shape)
            shape[1] = self.patch_num

            loss_avg = loss_avg.expand(shape)
            loss_avg = loss_avg.reshape(loss_ori.shape)
        print(loss_ori)
        print(loss_avg)
        loss_ori[loss_ori<=loss_avg]=0
        print(loss_ori)
        loss = loss_ori.sum()

        return loss

    def forward_loss(self, ori, res, tar):
        h_loss = self.forward_harmonization(res, tar)
        a_loss = self.forward_adversarial(ori, res)

        loss = h_loss * self.l_h + a_loss * self.l_a

        return loss

from PIL import Image
from torchvision import transforms

if __name__ == '__main__':

    topil = transforms.ToPILImage()
    topic = transforms.ToTensor()
    resize = transforms.Resize(size=(512, 512), interpolation=Image.BICUBIC)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss = Loss(device, nn.Module(), 1, 1)

    pil1 = Image.open("../ori.jpg")
    pil1 = resize(pil1)
    pic1 = topic(pil1).unsqueeze(0).to(device)

    pil2 = Image.open("../tar.jpg")
    pil2 = resize(pil2)
    pic2 = topic(pil2).unsqueeze(0).to(device)


    res = loss.forward_patch(pic1, pic2)
    print(res.grad)


nn.MSELoss