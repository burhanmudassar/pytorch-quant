import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from libs.utils import load_model
from libs.utils import print_model_stats
from libs.utils import savemodel_scripted
from libs.utils import quantization_post_dynamic
from libs.utils import quantization_post_dynamicx86
from libs.utils import quantization_qat

from PIL import Image
import numpy as np
import cv2


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

def prepare_data_loaders(data_path):

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            cv2Resize(256),
            transforms.CenterCrop(224),
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
    data_path = 'data/imagenet_1k'
    saved_model_dir = 'models/'
    float_model_file = 'mobilenet_pretrained_float.pth'

    train_batch_size = 30
    eval_batch_size = 10
    num_eval_batches = 100
    num_train_batches = 10
    num_calibration_batches = 10

    DEBUG_FLOAT_BENCHMARK = True
    DEBUG_DYNAMIC = True
    DEBUG_DYNAMIC_x86 = True
    DEBUG_QAT = True

    data_loader, data_loader_test = prepare_data_loaders(data_path)
    criterion = nn.CrossEntropyLoss()
    float_model = load_model(saved_model_dir + float_model_file)

    if DEBUG_FLOAT_BENCHMARK:
        float_model_scripted = savemodel_scripted(float_model, saved_model_dir + 'mobilenet_pretrained_float_scripted.pth')
        print_model_stats(float_model_scripted, criterion, data_loader_test, num_eval_batches=num_eval_batches, eval_batch_size=eval_batch_size)

    # dynamic
    if DEBUG_DYNAMIC:
        modelPostDynamic = quantization_post_dynamic(saved_model_dir + float_model_file, criterion, data_loader, data_loader_test, num_calibration_batches)
        print_model_stats(modelPostDynamic, criterion, data_loader_test, num_eval_batches=num_eval_batches, eval_batch_size=eval_batch_size)
        savemodel_scripted(modelPostDynamic, saved_model_dir + 'mobilenet_v2_int8_dynamic.pth')

    # x86 aware
    if DEBUG_DYNAMIC_x86:
        modelPostDynamic_x86 = quantization_post_dynamicx86(saved_model_dir + float_model_file, criterion, data_loader,
                                                    data_loader_test, num_calibration_batches)
        print_model_stats(modelPostDynamic_x86, criterion, data_loader_test, num_eval_batches=num_eval_batches, eval_batch_size=eval_batch_size)
        savemodel_scripted(modelPostDynamic_x86, saved_model_dir + 'mobilenet_v2_int8_dynamicx86.pth')

    # qat-aware training
    if DEBUG_QAT:
        modelQAT = quantization_qat(saved_model_dir + float_model_file, criterion, data_loader, data_loader_test, num_train_batches=num_train_batches, num_eval_batches=num_eval_batches)
        print_model_stats(modelQAT, criterion, data_loader_test, num_eval_batches=num_eval_batches, eval_batch_size=eval_batch_size)
        savemodel_scripted(modelQAT, saved_model_dir + 'mobilenet_v2_int8_qat.pth')

