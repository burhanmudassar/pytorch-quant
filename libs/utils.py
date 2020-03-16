import torch
import os
import time
from tqdm import tqdm
from libs.mobilenetv2 import MobileNetV2

cudaFlag = torch.cuda.is_available()
if cudaFlag:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5


def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def run_benchmark(model, img_loader):
    elapsed = 0
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    for cnt, (image, target) in enumerate(data_loader):
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    # print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
    #       .format(top1=top1, top5=top5))
    return

def savemodel_scripted(model, path):
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, path)
    return scripted_model

def print_model_stats(model, criterion, dataloader_test, num_eval_batches, eval_batch_size):
    # Accuracy
    top1, top5 = evaluate(model, criterion, dataloader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))

    # Runtime
    run_benchmark(model, dataloader_test)

    # Size
    print_size_of_model(model)

def quantization_post_dynamic(float_model_file, criterion, dataloader_train, dataloader_test, num_calibration_batches=10):
    myModel = load_model(float_model_file).to(device)
    myModel.eval()
    # Fuse Conv, bn and relu
    myModel.fuse_model()
    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    myModel.qconfig = torch.quantization.default_qconfig
    print(myModel.qconfig)
    torch.quantization.prepare(myModel, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)
    #
    # Calibrate with the training set
    evaluate(myModel, criterion, dataloader_train, neval_batches=num_calibration_batches)
    print('Post Training Quantization: Calibration done')
    #
    # # Convert to quantized model
    torch.quantization.convert(myModel, inplace=True)
    return myModel

def quantization_post_dynamicx86(float_model_file, criterion, dataloader_train, dataloader_test, num_calibration_batches=10):
    per_channel_quantized_model = load_model(float_model_file).to(device)
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
    per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')

    torch.quantization.prepare(per_channel_quantized_model, inplace=True)
    evaluate(per_channel_quantized_model, criterion, dataloader_train, num_calibration_batches)
    torch.quantization.convert(per_channel_quantized_model, inplace=True)
    return per_channel_quantized_model

def quantization_qat(float_model_file, criterion, dataloader_train, dataloader_test, num_train_batches=10, num_eval_batches=10):
    #Quantization-Aware Training
    qat_model =load_model(float_model_file)
    qat_model.fuse_model()

    optimizer = torch.optim.SGD(qat_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    qat_model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')

    torch.quantization.prepare_qat(qat_model, inplace=True)
    print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',
          qat_model.features[1].conv)

    quantized_model = None
    for _, nepoch in enumerate(tqdm(range(8))):
        train_one_epoch(qat_model, criterion, optimizer, dataloader_train, device, num_train_batches)
        if nepoch > 3:
            qat_model.apply(torch.quantization.disable_observer)
        if nepoch > 2:
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        scheduler.step()

        quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
        quantized_model.eval()

        top1, top5 = evaluate(quantized_model, criterion, dataloader_test, neval_batches=num_eval_batches)
        print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * dataloader_test.batch_size, top1.avg))

    return quantized_model