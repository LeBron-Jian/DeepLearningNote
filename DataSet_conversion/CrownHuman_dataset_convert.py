from xml.dom import minidom
import cv2
import os
import json
from PIL import Image

roadlabels = "/datasets/CrowdHuman_dataset/Annotations_vbody/"
roadimages = "/datasets/CrowdHuman_dataset/Images/"
fpath = "/datasets/CrowdHuman_dataset/annotation_val.odgt"


def load_func(fpath):
    assert os.path.exists(fpath)
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records


bbox = load_func(fpath)
# bbox中的其中一行如下
# {'ID': '284193,faa9000f2678b5e',
# 'gtboxes': [{'tag': 'person', 'hbox': [123, 129, 63, 64], 'head_attr': {'ignore': 0, 'occ': 1, 'unsure': 0}, 'fbox': [61, 123, 191, 453], 'vbox': [62, 126, 154, 446], 'extra': {'box_id': 0, 'occ': 1}},
#             {'tag': 'person', 'hbox': [214, 97, 58, 74], 'head_attr': {'ignore': 0, 'occ': 1, 'unsure': 0}, 'fbox': [165, 95, 187, 494], 'vbox': [175, 95, 140, 487], 'extra': {'box_id': 1, 'occ': 1}},
#             {'tag': 'person', 'hbox': [318, 109, 58, 68], 'head_attr': {'ignore': 0, 'occ': 1, 'unsure': 0}, 'fbox': [236, 104, 195, 493], 'vbox': [260, 106, 170, 487], 'extra': {'box_id': 2, 'occ': 1}},
#             {'tag': 'person', 'hbox': [486, 119, 61, 74], 'head_attr': {'ignore': 0, 'occ': 0, 'unsure': 0}, 'fbox': [452, 110, 169, 508], 'vbox': [455, 113, 141, 501], 'extra': {'box_id': 3, 'occ': 1}},
#             {'tag': 'person', 'hbox': [559, 105, 53, 57], 'head_attr': {'ignore': 0, 'occ': 0, 'unsure': 0}, 'fbox': [520, 95, 163, 381], 'vbox': [553, 98, 70, 118], 'extra': {'box_id': 4, 'occ': 1}},
#             {'tag': 'person', 'hbox': [596, 40, 72, 83], 'head_attr': {'ignore': 0, 'occ': 0, 'unsure': 0}, 'fbox': [546, 39, 202, 594], 'vbox': [556, 39, 171, 588], 'extra': {'box_id': 5, 'occ': 1}},
#             {'tag': 'person', 'hbox': [731, 139, 69, 83], 'head_attr': {'ignore': 0, 'occ': 0, 'unsure': 0}, 'fbox': [661, 132, 183, 510], 'vbox': [661, 132, 183, 510], 'extra': {'box_id': 6, 'occ': 0}}]}

if not os.path.exists(roadlabels):
    os.makedirs(roadlabels)

for i0, item0 in enumerate(bbox):
    print(i0)
    # 建立i0的xml tree
    ID = item0['ID']  # 得到当前图片的名字
    imagename = roadimages + ID + '.jpg'  # 当前图片的完整路径
    savexml = roadlabels + ID + '.xml'  # 生成的.xml注释的名字

    # 获得图片的长宽
    # img = Image.open(imagename)
    # img_width = img.size[0]
    # img_height = img.size[1]

    gtboxes = item0['gtboxes']
    img_name = ID
    floder = 'CrowdHuman'
    print(imagename)
    im = cv2.imread(imagename)
    w = im.shape[1]
    h = im.shape[0]
    d = im.shape[2]

    doc = minidom.Document()  # 创建DOM树对象
    annotation = doc.createElement('annotation')  # 创建子节点
    doc.appendChild(annotation)  # annotation作为doc树的子节点

    folder = doc.createElement('folder')
    folder.appendChild(doc.createTextNode(floder))  # 文本节点作为floder的子节点
    annotation.appendChild(folder)  # folder作为annotation的子节点

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode(img_name + '.jpg'))
    annotation.appendChild(filename)

    # filename = doc.createElement('path')
    # filename.appendChild(doc.createTextNode('D:/BaiduNetdiskDownload/CrowdHuman_train/Images'))
    # annotation.appendChild(filename)

    source = doc.createElement('source')
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode("Unknown"))
    source.appendChild(database)
    # annotation2 = doc.createElement('annotation')
    # annotation2.appendChild(doc.createTextNode("ICDAR POD2017"))
    # source.appendChild(annotation2)
    # image = doc.createElement('image')
    # image.appendChild(doc.createTextNode("image"))
    # source.appendChild(image)
    # flickrid = doc.createElement('flickrid')
    # flickrid.appendChild(doc.createTextNode("NULL"))
    # source.appendChild(flickrid)
    annotation.appendChild(source)

    # owner = doc.createElement('owner')
    # flickrid = doc.createElement('flickrid')
    # flickrid.appendChild(doc.createTextNode("NULL"))
    # owner.appendChild(flickrid)
    # na = doc.createElement('name')
    # na.appendChild(doc.createTextNode("cxm"))
    # owner.appendChild(na)
    # annotation.appendChild(owner)

    size = doc.createElement('size')
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode("%d" % w))
    size.appendChild(width)
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode("%d" % h))
    size.appendChild(height)
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode("%d" % d))
    size.appendChild(depth)
    annotation.appendChild(size)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode("0"))
    annotation.appendChild(segmented)

    # 下面是从odgt中提取三种类型的框并转为voc格式的xml的代码
    # 不需要的box种类整段注释即可
    for i1, item1 in enumerate(gtboxes):

        # 提取可见框(visible box)的代码
        boxs = [int(a) for a in item1['vbox']]
        minx = str(boxs[0])
        miny = str(boxs[1])
        maxx = str(boxs[2] + boxs[0])
        maxy = str(boxs[3] + boxs[1])
        # print(box)
        object = doc.createElement('object')
        nm = doc.createElement('name')
        # nm.appendChild(doc.createTextNode('vbox'))
        nm.appendChild(doc.createTextNode('person'))
        object.appendChild(nm)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode("Unspecified"))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode("1"))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode("0"))
        object.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(minx))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(miny))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(maxx))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(maxy))
        bndbox.appendChild(ymax)
        object.appendChild(bndbox)
        annotation.appendChild(object)
        savefile = open(savexml, 'w')
        savefile.write(doc.toprettyxml())
        savefile.close()
        '''
        # 提取头部框(head box)的代码
        boxs = [int(a) for a in item1['hbox']]
        minx = str(boxs[0])
        miny = str(boxs[1])
        maxx = str(boxs[2] + boxs[0])
        maxy = str(boxs[3] + boxs[1])
        # print(box)
        object = doc.createElement('object')
        nm = doc.createElement('name')
        nm.appendChild(doc.createTextNode('hbox'))
        object.appendChild(nm)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode("Unspecified"))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode("1"))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode("0"))
        object.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(minx))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(miny))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(maxx))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(maxy))
        bndbox.appendChild(ymax)
        object.appendChild(bndbox)
        annotation.appendChild(object)
        savefile = open(savexml, 'w')
        savefile.write(doc.toprettyxml())
        savefile.close()

        # 提取全身框(full box)的标注
        boxs = [int(a) for a in item1['fbox']]
        # 左上点长宽--->左上右下
        minx = str(boxs[0])
        miny = str(boxs[1])
        maxx = str(boxs[2] + boxs[0])
        maxy = str(boxs[3] + boxs[1])
        # print(box)
        object = doc.createElement('object')
        nm = doc.createElement('name')
        nm.appendChild(doc.createTextNode('fbox'))  # 类名: fbox
        object.appendChild(nm)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode("Unspecified"))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode("1"))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode("0"))
        object.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(minx))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(miny))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(maxx))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(maxy))
        bndbox.appendChild(ymax)
        object.appendChild(bndbox)
        annotation.appendChild(object)
        savefile = open(savexml, 'w')
        savefile.write(doc.toprettyxml())
        savefile.close()
        '''
