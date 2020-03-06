import glob
import os
import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':
    impaths = []
    labels = []
    with open('labels.txt', 'r') as fid:
        for line in fid.readlines():
            im_folder, label = line.split(',')
            im_file = glob.glob(os.path.join('val', im_folder, '*.JPEG'))
            impaths.append(im_file)
            labels.append(label)

    with open('../labels_converted.txt', 'w') as fid:
        for idx, (imgpath, label) in enumerate(zip(impaths, labels)):
            fid.write('{},{:d}\n'.format(imgpath[0], idx+1))

    # dataset_test = torchvision.datasets.ImageFolder(
    #     'val',
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #     ]))

    # pass