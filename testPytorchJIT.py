import torch
import cv2
import torchvision
import os
from torchvision.transforms import transforms
from libs.utils import AverageMeter
from libs.utils import accuracy
from PIL import Image
import numpy as np

class cv2Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im):
        im = np.array(im)
        if isinstance(self.size, int):
            h, w = im.shape[:2]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return Image.fromarray(im)
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return Image.fromarray(cv2.resize(im, (ow, oh)))
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return Image.fromarray(cv2.resize(im, (ow, oh)))
        else:
            return Image.fromarray(cv2.resize(im, self.size[::-1]))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def prepare_data_loaders(data_path):

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            cv2Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            cv2Resize(256),
            transforms.CenterCrop(224),
            # cv2Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test

if __name__ == '__main__':
    train_batch_size = 1
    eval_batch_size = 1

    data_path = 'data/imagenet_1k'
    # modelpath = 'data/mobilenet_v2_fp32_scripted.pth'
    modelpath = 'data/mobilenet_v2_int8_static_qnnpack.pth'

    scripted_model = torch.jit.load(modelpath)
    data_loader, data_loader_test = prepare_data_loaders(data_path)

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    for cnt, (image, target) in enumerate(data_loader_test):
        output = scripted_model(image)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        print('.', end='')
        top1.update(acc1[0], image.shape[0])
        top5.update(acc5[0], image.shape[0])
        if cnt >= 1000:
            break

    print('\nEvaluation accuracy on %d images, %2.3f %2.3f' % (len(data_loader_test), top1.avg, top5.avg))

    # testIm = cv2.imread('fox.jpg')
    # testIm = testIm[:, :, ::-1]

    # output = scripted_model(testImTensor)





