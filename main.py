import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.quantization import QuantStub, DeQuantStub

from libs.utils import load_model
from libs.utils import print_size_of_model
from libs.utils import evaluate
from libs.utils import run_benchmark

from torchvision.models.utils import load_state_dict_from_url


# Fix for windows quantization
# torch.backends.quantized.engine = 'qnnpack'

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
            transforms.Resize(256),
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
    saved_model_dir = 'data/'
    float_model_file = 'mobilenet_pretrained_float.pth'
    scripted_float_model_file = 'mobilenet_v2_fp32_scripted.pth'
    scripted_quantized1_model_file = 'mobilenet_v2_int8_dynamic.pth'
    scripted_quantized2_model_file = 'mobilenet_v2_int8_static_qnnpack.pth'

    train_batch_size = 30
    eval_batch_size = 10
    num_eval_batches = 100
    num_calibration_batches = 5

    data_loader, data_loader_test = prepare_data_loaders(data_path)
    criterion = nn.CrossEntropyLoss()
    float_model = load_model(saved_model_dir + float_model_file).to('cpu')

    # Baseline EVAL
    # print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
    # float_model.eval()
    #
    # # Fuses modules
    # float_model.fuse_model()
    #
    # # Note fusion of Conv+BN+Relu and Conv+Relu
    # print('\n Inverted Residual Block: After fusion\n\n', float_model.features[1].conv)
    #
    # ### Baseline accuracy
    #
    # print("Size of baseline model")
    # print_size_of_model(float_model)
    #
    # top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    # print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))
    # torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)
    #
    #
    # # Quantization - Post
    num_calibration_batches = 30
    #
    # myModel = load_model(saved_model_dir + float_model_file).to('cpu')
    # myModel.eval()
    # # #
    # # # # Fuse Conv, bn and relu
    # myModel.fuse_model()
    # #
    # # # # Specify quantization configuration
    # # # # Start with simple min/max range estimation and per-tensor quantization of weights
    # myModel.qconfig = torch.quantization.default_qconfig
    # print(myModel.qconfig)
    # torch.quantization.prepare(myModel, inplace=True)
    #
    # # # Calibrate first
    # # print('Post Training Quantization Prepare: Inserting Observers')
    # # print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)
    # #
    # # # Calibrate with the training set
    # # evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
    # # print('Post Training Quantization: Calibration done')
    # #
    # # # Convert to quantized model
    # # torch.quantization.convert(myModel, inplace=True)
    # # print('Post Training Quantization: Convert done')
    # # print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',
    # #       myModel.features[1].conv)
    # #
    # # print("Size of model after quantization")
    # # print_size_of_model(myModel)
    # #
    # # top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
    # # print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))
    # # torch.jit.save(torch.jit.script(myModel), saved_model_dir + scripted_quantized1_model_file)
    #
    #
    # Quantization - x86 Aware
    per_channel_quantized_model = load_model(saved_model_dir + float_model_file).to('cpu')
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
    per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    print(per_channel_quantized_model.qconfig)

    torch.quantization.prepare(per_channel_quantized_model, inplace=True)
    evaluate(per_channel_quantized_model, criterion, data_loader, num_calibration_batches)
    torch.quantization.convert(per_channel_quantized_model, inplace=True)
    top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))
    torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized2_model_file)
    print("Size of model after quantization")
    print_size_of_model(per_channel_quantized_model)

    # Do benchmarking
    run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)
    # run_benchmark(saved_model_dir + scripted_quantized1_model_file, data_loader_test)
    run_benchmark(saved_model_dir + scripted_quantized2_model_file, data_loader_test)