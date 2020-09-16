import tensorflow as tf
from random import shuffle
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K


class MultiboxLoss(object):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported')
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard

    def _l1_smooth_loss(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred),
                                      axis=-1)
        return softmax_loss

    def compute_loss(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.to_float(tf.shape(y_true)[1])

        # 计算所有的loss
        # 分类的loss
        # batch_size,8732,21 -> batch_size,8732
        conf_loss = self._softmax_loss(y_true[:, :, 4:-8],
                                       y_pred[:, :, 4:-8])
        # 框的位置的loss
        # batch_size,8732,4 -> batch_size,8732
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])

        # 获取所有的正标签的loss
        # 每一张图的pos的个数
        num_pos = tf.reduce_sum(y_true[:, :, -8], axis=-1)
        # 每一张图的pos_loc_loss
        pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -8],
                                     axis=1)
        # 每一张图的pos_conf_loss
        pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -8],
                                      axis=1)

        # 获取一定的负样本
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos,
                             num_boxes - num_pos)

        # 找到了哪些值是大于0的
        pos_num_neg_mask = tf.greater(num_neg, 0)
        # 获得一个1.0
        has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
        num_neg = tf.concat( axis=0,values=[num_neg,
                                [(1 - has_min) * self.negatives_for_hard]])
        # 求平均每个图片要取多少个负样本
        num_neg_batch = tf.reduce_mean(tf.boolean_mask(num_neg,
                                                      tf.greater(num_neg, 0)))
        num_neg_batch = tf.to_int32(num_neg_batch)

        # conf的起始
        confs_start = 4 + self.background_label_id + 1
        # conf的结束
        confs_end = confs_start + self.num_classes - 1

        # 找到实际上在该位置不应该有预测结果的框，求他们最大的置信度。
        max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end],
                                  axis=2)
        
        # 取top_k个置信度，作为负样本
        _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]),
                                 k=num_neg_batch)

        # 找到其在1维上的索引
        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
        full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) +
                        tf.reshape(indices, [-1]))
        

        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]),
                                  full_indices)
        neg_conf_loss = tf.reshape(neg_conf_loss,
                                   [batch_size, num_neg_batch])
        neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

        # 求loss总和
        total_loss = pos_conf_loss + neg_conf_loss
        total_loss /= (num_pos + tf.to_float(num_neg_batch))
        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,
                            tf.ones_like(num_pos))
        total_loss += (self.alpha * pos_loc_loss) / num_pos
        return total_loss

class Generator(object):
    def __init__(self, bbox_util,batch_size,
                 train_lines, val_lines, image_size,num_classes,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3./4., 4./3.]):
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.val_lines = val_lines
        self.train_batches = len(train_lines)
        self.val_batches = len(val_lines)
        self.image_size = image_size
        self.color_jitter = []
        self.num_classes = num_classes
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range
        
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
    
    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y
    
    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y
    
    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))     
        h = np.round(np.sqrt(target_area / random_ratio))
        
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        h = min(h, img_h)
        
        w = min(h,w)
        h = w
        
        w_rel = w / img_w
        w = int(w)
        h_rel = h / img_h
        h = int(h)
        
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y+h, x:x+w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets
    
    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_lines)
                lines = self.train_lines
            else:
                shuffle(self.val_lines)
                lines = self.val_lines
            inputs = []
            targets = []
            for annotation_line in lines:  
                line = annotation_line.split()
                # 读取图片
                img_path = line[0]
                img = np.array(Image.open(img_path),dtype=np.float32)
                
                # 对txt读取进来的框的位置与class进行处理
                shape = np.shape(img)
                y = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
                if len(y)==0:
                    continue
                boxes = np.array(y[:,:4],dtype=np.float32)
                boxes[:,0] = boxes[:,0]/shape[1]
                boxes[:,1] = boxes[:,1]/shape[0]
                boxes[:,2] = boxes[:,2]/shape[1]
                boxes[:,3] = boxes[:,3]/shape[0]
                one_hot_label = np.eye(self.num_classes-1)[y[:,4]]
                y = np.concatenate([boxes,one_hot_label],axis=-1)

                # 随机裁剪
                if self.do_crop:
                    img, y = self.random_sized_crop(img, y)

                img = np.array(Image.fromarray(np.uint8(img)).resize(self.image_size),dtype=np.float32)
                # 在训练的时候，可以随机增加噪声
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                # 计算真实框对应的先验框，与这个先验框应当有的预测结果
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)                
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets
