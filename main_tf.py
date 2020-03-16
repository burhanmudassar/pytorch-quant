import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import tensorflow as tf
from libs.utils import accuracy, AverageMeter
import cv2

def detection_collate(batch):
    image = []
    target = []
    for sample in batch:
        image.append(np.array(sample[0]))
        target.append([sample[1]])

    return np.array(image), np.array(target)


def prepare_data_loaders(data_path):

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.ToTensor(),
        ]))

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.ToTensor(),
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler, collate_fn=detection_collate)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler, collate_fn=detection_collate)

    return data_loader, data_loader_test

if __name__ == '__main__':
    train_batch_size = 1
    eval_batch_size = 1
    data_path = 'data/imagenet_1k'
    saved_model_dir = 'data/'
    float_model_file = 'models_old/mobilenet_v2_1.0_224_quant.tflite'

    input_mean = 127.5
    input_std = 127.5

    interpreter = tf.lite.Interpreter(model_path=float_model_file)
    interpreter.allocate_tensors()

    data_loader, data_loader_test = prepare_data_loaders(data_path)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model = input_details[0]['dtype'] == np.float32

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    for image, target in data_loader_test:
        image_resized = []
        for im in image:
            image_resized += [cv2.resize(im, (width, height))]
        image = np.asarray(image_resized)

        if floating_model:
            input_data = (np.float32(image) - input_mean) / input_std
        else:
            input_data = image.astype(np.uint8)


        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        cnt += 1
        acc1, acc5 = accuracy(torch.from_numpy(output_data), torch.from_numpy(target+1), topk=(1, 5))
        print('.', end='')
        top1.update(acc1[0], image.shape[0])
        top5.update(acc5[0], image.shape[0])
        if cnt >= 1000:
            break

    print('\nEvaluation accuracy on %d images, %2.3f %2.3f' % (len(data_loader_test), top1.avg, top5.avg))

