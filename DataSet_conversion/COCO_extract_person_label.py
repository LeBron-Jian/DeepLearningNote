import json
import os
import shutil
import skimage.io as io
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image, ImageDraw

from voc_format import headstr, tailstr, objstr


def mkr(path):
    # 检查目录是否存在，如果存在，先删除再创建，否则，直接创建
    if not os.path.exists(path):
        # 可以创建多级目录
        os.makedirs(path)


def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


def write_xml(anno_path, head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr % (obj[0], obj[1], obj[2], obj[3], obj[4]))
    f.write(tail)


def save_annotations_and_imgs(coco, image_dir, save_image_path, save_label_path, filename, objs):
    # 将图片转为xml，例:COCO_train2017_000000196610.jpg-->COCO_train2017_000000196610.xml
    mkr(save_label_path)
    anno_path = save_label_path + '/' + filename[:-3]+'xml'
    img_path = image_dir + '/'+filename

    mkr(save_image_path)
    dst_imgpath = save_image_path + '/' + filename

    img = cv2.imread(img_path)
    # if (img.shape[2] == 1):
    #    print(filename + " not a RGB image")
    #   return
    shutil.copy(img_path, dst_imgpath)

    head = headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path, head, objs, tail)


def showimg(coco, dataset, img, classes, cls_id, show=False):
    I = Image.open('%s/%s' % (dataset, img['file_name']))
    # 通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name = classes[ann['category_id']]
        if class_name in classes_names:
            # print(class_name)
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()

    return objs


if __name__ == '__main__':
    # json file name
    jsonfilename = '/datasets/COCO2017_val/JSON_folder/instances_train2017.json'
    # image_dir
    image_dir = '/datasets/COCO2017_tovoc/train2017/'
    # save path;  label format is VOC format(.xml)
    save_image_path = '/datasets/COCO2017_train/person_image/'
    save_label_path = 'datasets/COCO2017_train/annotations/'
    # 提取COCO 数据集中特定的类别， 这里全部提取人
    classes_names = ['person']

    # 使用COCO API用来初始化注释数据
    coco = COCO(jsonfilename)

    # 获取COCO数据集中的所有类别
    classes = id2name(coco)
    #[1, 2, 3, 4, 6, 8]
    classes_ids = coco.getCatIds(catNms=classes_names)
    for cls in classes_names:
        # 获取该类的id
        cls_id = coco.getCatIds(catNms=[cls])
        img_ids = coco.getImgIds(catIds=cls_id)
        # print(cls, len(img_ids))
        # imgIds=img_ids[0:10]
        for imgId in tqdm(img_ids):
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            # print(filename)
            objs = showimg(coco, image_dir, img, classes, classes_ids, show=False)
            print(objs)
            save_annotations_and_imgs(coco, image_dir, save_image_path,
                                      save_label_path, filename, objs)
