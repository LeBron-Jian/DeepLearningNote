"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modifications Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn

# from .common import inverse_sigmoid
# from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

from common import inverse_sigmoid
from box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0, ):
    """cnd
    此函数是用于训练时生成对比学习风格的噪声样本组Denoising group，以增强 Transformer的鲁棒性和收敛速度，特别是在目标检测中的查询学习
    Query-based Detection 任务中，比如DN-DETR，RT-DETR, DFINE等架构。
    总的说，该函数的目的是构建一个denoising训练组（正负混合），将带噪声的分类+框信息喂给Transformer Decoder，用于提升网络对Query预测的鲁棒性
    避免decoder在训练初期陷入随机状态
    targets	每张图像的标注信息，如 boxes, labels，即真实标签
    num_classes	类别总数
    num_queries	原始 decoder 的 query 数量
    class_embed	分类的 embedding 层
    num_denoising	denoising queries 的数量上限 100
    label_noise_ratio	分类扰动的比例，即类别噪声的比例，此处设置为0.5，表示有一半的类别要改变
    box_noise_scale	边界框扰动的幅度

                       GT                              Noisy GT
             +-------------+                 +-------------------+
             | labels/box  |      tile →     | 2x groups (pos/neg)|
             +-------------+                 +-------------------+
                        \                         /
                         \ add noise (随机扰动)
                          ↓
                 ┌─────────────────────────────┐
                 │ input_query_class / bbox    │
                 │ attn_mask (group互不可见)    │
                 │ dn_meta（训练辅助信息）      │
                 └─────────────────────────────┘

        这个函数做了下面几件事：
        1，复制每张图像的GT多次---> 构成 deboising group
        2，加入分类扰动与框扰动----> 模拟真实分布外样本
        3，构造 attention mask--> 避免 denoising 样本间干扰
        4，输出logits和unactivated bbox---> 供Transformer Decoder 使用
    """
    if num_denoising <= 0:
        return None, None, None, None
    # 统计ground truth 。首先确定每张图像中标注样本的个数，保存在num_gts中，
    num_gts = [len(t['labels']) for t in targets]  # 每张图像中的GT数量
    device = targets[0]['labels'].device

    max_gt_num = max(num_gts)  # 随后选出当前 batch 中最多的GT， 因为方便在后面划分组时实用，默认num_denoising的数量为100
    if max_gt_num == 0:
        return None, None, None, None

    # 我们在每个batch（批次）内划分创建噪声，因此，噪声组的数目为 num_denoising // max_gt_num。
    num_group = num_denoising // max_gt_num
    # print("num_group, num_denoising, max_gt_num is ", num_group, num_denoising, max_gt_num)  #  11 100 9
    num_group = 1 if num_group == 0 else num_group  # 确保num_group至少为1，避免为0导致后续出错
    # pad gt to max_num of a batch
    bs = len(num_gts)

    # 随后生成对应的输入类别，输出标注框以及真值掩膜，
    # input_query_class 创建一个二维的张量，并用一个指定的值num_classes来填充它，其中[bs, max_gt_num]就是创建张量的形状或尺寸
    # 如果模型可以检测80种不同的对象类别（ID为0~79），那么num_classes可能是80或81，如果num_classes被用来填充，它就代表一个背景或占位符类别，比任何实际的对象类别ID都大
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)
    # print("input_query_class shape is ", input_query_class.shape)  # torch.Size([4, 9])  4表示batch大小，9表示该batch内图像中标注样本的最大值
    # print("input_query_bbox shape is ", input_query_bbox.shape)  # torch.Size([4, 9, 4]) 最后一个4表示w,h,x,y
    # print("pad_gt_mask shape is ", pad_gt_mask.shape)  # torch.Size([4, 9])  真值掩膜

    # 为其赋予对应的值
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1

    # 随后开始创建噪声了，要分正负样本，还要分num_group，所以是2*num_group
    # each group has positive and negative queries.
    # tile函数的作用是沿着指定的维度重复张量，它主要接受两个参数，
    # 1输入张量input_query_class，你想要重复的原始张量
    # 2.重复次数[1, 2*num_group]一个列表或元素，指定了在每个维度上重复的次数，这个列表的长度必须和输入张量的维度数量rank一致
    # 比如input_query_class是一个二维张量，形状是[1, 2 * num_group], 1表示在第一个维度（行维度，索引为0），张量不进行重复，即重复1次， 2*num_group表示在第二个维度上，张量会重复2*num_group次
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # print("input_query_class shape is ", input_query_class.shape)  # torch.Size([4, 9]) --》torch.Size([4, 198])
    # print("input_query_bbox shape is ", input_query_bbox.shape)  # torch.Size([4, 9, 4])--》torch.Size([4, 198, 4])
    # print("pad_gt_mask shape is ", pad_gt_mask.shape)  # torch.Size([4, 9]) --》 torch.Size([4, 198])

    # positive and negative mask
    # max_gt_num * 2: 这是一个关键点。它表明对于每个真实目标，会生成两倍数量的“查询”或“占位符”；
    # 前max_gt_num个位置可能对应原始的真实目标，后max_gt_num个位置可能对应通过添加噪声生成的“负样本”或“填充”。
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    # 将张量的后半部分（从max_gt_num索引开始）全部设置为1，明确的标记了那些不是直接对应原始真实目标的位置
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    # 这是一个简单的逐元素减法，得到的positive_gt_mask，所有原来为0的位置变为1，所有原来为1的位置变为0
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    # 将正负样本掩膜与标注位置掩膜结合，并获取每个正样本的坐标
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    # 添加类别噪声
    if label_noise_ratio > 0:
        # 首先是生成噪声比例的掩膜mask，创建一个随机的二进制掩码（binary mask），这个掩码用于决定在标签class信息中引入噪声的哪些位置
        # rand_like 创建一个与输入张量形状相同，数据类型指定的张量，并在[0,1)均匀分布中随机抽取的浮点数填充
        # label_noise_ratio是一个预定义的超参数，代表希望引入标签噪声的总比例，乘以0.5是一个常见做法，特别是去噪训练中，它通常意味着标签噪声被分为了两个部分
        # 一部分用于将真实标签随机替换为其他类别，另一部分是用于将真实标签替换为“未知”或“背景”类别，例如num_classes
        # 最终得到一个布尔类型的mask，形状与input_query_class相同，在这个 mask 中，大约有 (label_noise_ratio * 0.5) 比例的元素是 True，其余为 False
        # mask中为true的位置，这些位置的标签将被选中，并进行后续的噪声处理；mask为False的位置，这些位置的标签将保持不变，或者进行另一种类型的噪声处理
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here 随后根据掩膜生成随机类别new_label， 这一步为那种注入噪声的位置，随机生成一个“新的”类别标签，这个新的标签可以是任意一个有效的类别ID
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        # 这是核心的标签噪声注入逻辑，pad_gt_mask是一个非常关键的掩码，通常表示哪些位置对应有效的真实目标Ground truth，而不是填充padding
        # mask & pad_gt_mask的结果只有当一个位置同时被mask选中（随机噪声概率），该位置对应的是一个有效的真实目标查询（非填充，非负样本），这个复合条件为True
        # 最后判断input_query_class对应位置是不是应该添加噪声，最终生成噪声input_query_class
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    # 添加标注框噪声
    if box_noise_scale > 0:
        # known box: [cx, cy, w, h] → [x1, y1, x2, y2]  首先进行一个转换，原本的标注文件给的中心点坐标与宽高的标注个数。我们将其转变为左上右下坐标，方便添加噪声
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        # diff是根据目标框的宽高获取一个缩放的比例，其中使用宽高乘以0.5， 能够保证将来中心点偏移不会超出原本的标注框
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        # rand_sign 是目标框位移的方向， torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0生成的值在-1到1之间，将其作用在坐标上，即可实现上下左右的平移
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        # 随后对正样本以及负样本添加不同程度的噪声区别
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)  # 负样本的噪声更大
        known_bbox += (rand_sign * rand_part * diff)  # 扰动值：将偏移方向rand_sign与缩放程度diff相乘可以得到两个坐标点偏移后的位置
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        # 最终将偏移后的坐标转换为中心点宽高形式，并构造查询向量
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        # FIXME, RT-DETR do not have this 
        input_query_bbox[input_query_bbox < 0] *= -1
        # 函数 inverse_sigmoid 的作用： 将 normalized [0, 1] box 映射到未激活的形式（Decoder 预测的是这个）。
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    input_query_logits = class_embed(input_query_class)

    # 遮蔽掩膜设计：是为了让真值加噪生成的查询向量与Encoder输入的查询向量有所区分，因为如果Decoder对上述不区分的话，虽然加噪组添加了噪声，但是相比Encoder输出的查询向量也是十分强的
    # 这就会导致作弊，因此加噪组查询向量与原始查询向量需要加以区分，而这个做法便是遮蔽掩膜，其实这个掩膜与前面噪声构造时的很相似；同时不同的噪声组之间也是需要相互屏蔽的
    # 加噪的与不加噪的不可见
    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other  加噪组之间不可见
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])

    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta


if __name__ == "__main__":
    from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

    targets = [{'boxes': BoundingBoxes([[0.0000, 50.1197, 608.9715, 480.4698],
                                        [222.9757, 162.8899, 636.2614, 628.0797]], device='cpu',
                                       format=BoundingBoxFormat.XYXY, canvas_size=(640, 640)),
                'labels': torch.tensor([25, 0], device='cpu'), 'image_id': torch.tensor([1], device='cpu'),
                'area': torch.tensor([196962.6875, 144492.6250], device='cpu'),
                'iscrowd': torch.tensor([0, 0], device='cpu'),
                'orig_size': torch.tensor([481, 640], device='cpu'),
                'idx': torch.tensor([0], device='cpu')},

               {'boxes': BoundingBoxes([[261.2003, 270.0592, 309.2803, 345.1280],
                                        [393.3900, 275.8559, 425.9200, 327.9739],
                                        [370.1101, 326.4259, 453.2000, 388.9046],
                                        [223.2499, 328.0000, 365.1699, 429.9149],
                                        [261.5401, 257.7968, 295.7501, 328.8528]], device='cpu',
                                       format=BoundingBoxFormat.XYXY, canvas_size=(640, 640)),
                'labels': torch.tensor([0, 0, 20, 20, 0], device='cpu'),
                'image_id': torch.tensor([4], device='cpu'),
                'area': torch.tensor([2752.0972, 1292.7411, 3958.4058, 11028.6172, 1853.5038],
                                     device='cpu'), 'iscrowd': torch.tensor([0, 0, 0, 0, 0], device='cpu'),
                'orig_size': torch.tensor([640, 488], device='cpu'),
                'idx': torch.tensor([3], device='cpu')},

               {'boxes': BoundingBoxes([[214.1498, 55.2838, 562.4096, 381.6839]], device='cpu',
                                          format=BoundingBoxFormat.XYXY, canvas_size=(640, 640)),
                'labels': torch.tensor([16], device='cpu'), 'image_id': torch.tensor([2], device='cpu'),
                'area': torch.tensor([84898.7812], device='cpu'), 'iscrowd': torch.tensor([0], device='cpu'),
                'orig_size': torch.tensor([640, 478], device='cpu'),
                'idx': torch.tensor([1], device='cpu')},

               {'boxes': BoundingBoxes([[273.0838, 289.9968, 492.1450, 526.0672],
                                        [137.2387, 313.5232, 277.8538, 516.0192],
                                        [341.6522, 333.3504, 451.9978, 426.8416],
                                        [198.9376, 334.4896, 294.5344, 415.0400],
                                        [200.4664, 427.7888, 220.5231, 470.7584],
                                        [338.9312, 545.5232, 457.9277, 605.9776],
                                        [477.6312, 426.7520, 495.9073, 461.8624],
                                        [320.8401, 427.9936, 332.8674, 457.9584],
                                        [582.4841, 426.7520, 607.3450, 438.5536]], device='cpu',
                                       format=BoundingBoxFormat.XYXY, canvas_size=(640, 640)),
                'labels': torch.tensor([17, 17, 0, 0, 0, 58, 0, 0, 0], device='cpu'),
                'image_id': torch.tensor([3], device='cpu'),
                'area': torch.tensor([24051.4844, 13242.9043, 4798.0059, 3581.3445, 400.8260, 3345.7776,
                                      298.4381, 167.6160, 136.4558], device='cpu'),
                'iscrowd': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], device='cpu'),
                'orig_size': torch.tensor([381, 500], device='cpu'),
                'idx': torch.tensor([2], device='cpu')}]


    input_query_logits, input_query_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
        targets,
        num_classes=80,
        num_queries=300,
        class_embed=nn.Embedding(81, 128, padding_idx=80),
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0, )
    print('\n\n\n')
    print("input_query_logits is ", input_query_logits.shape)
    print("input_query_bbox_unact is ", input_query_bbox_unact.shape)
    print("attn_mask is ", attn_mask.shape)
    # print("dn_meta is ", dn_meta['dn_positive_idx'][0].shape, dn_meta[''])

    # targets is [{'boxes': BoundingBoxes([[0.0000, 50.1197, 608.9715, 480.4698],
    #                                      [222.9757, 162.8899, 636.2614, 628.0797]], device='cpu',
    #                                     format=BoundingBoxFormat.XYXY, canvas_size=(640, 640)),
    #              'labels': tensor([25, 0], device='cpu'), 'image_id': tensor([1], device='cpu'),
    #              'area': tensor([196962.6875, 144492.6250], device='cpu'),
    #              'iscrowd': tensor([0, 0], device='cpu'), 'orig_size': tensor([481, 640], device='cpu'),
    #              'idx': tensor([0], device='cpu')},
    #             {'boxes': BoundingBoxes([[261.2003, 270.0592, 309.2803, 345.1280],
    #                                      [393.3900, 275.8559, 425.9200, 327.9739],
    #                                      [370.1101, 326.4259, 453.2000, 388.9046],
    #                                      [223.2499, 328.0000, 365.1699, 429.9149],
    #                                      [261.5401, 257.7968, 295.7501, 328.8528]], device='cpu',
    #                                     format=BoundingBoxFormat.XYXY, canvas_size=(640, 640)),
    #              'labels': tensor([0, 0, 20, 20, 0], device='cpu'), 'image_id': tensor([4], device='cpu'),
    #              'area': tensor([2752.0972, 1292.7411, 3958.4058, 11028.6172, 1853.5038],
    #                             device='cpu'), 'iscrowd': tensor([0, 0, 0, 0, 0], device='cpu'),
    #              'orig_size': tensor([640, 488], device='cpu'), 'idx': tensor([3], device='cpu')}, {
    #                 'boxes': BoundingBoxes([[214.1498, 55.2838, 562.4096, 381.6839]], device='cpu',
    #                                        format=BoundingBoxFormat.XYXY, canvas_size=(640, 640)),
    #                 'labels': tensor([16], device='cpu'), 'image_id': tensor([2], device='cpu'),
    #                 'area': tensor([84898.7812], device='cpu'), 'iscrowd': tensor([0], device='cpu'),
    #                 'orig_size': tensor([640, 478], device='cpu'), 'idx': tensor([1], device='cpu')},
    #             {'boxes': BoundingBoxes([[273.0838, 289.9968, 492.1450, 526.0672],
    #                                      [137.2387, 313.5232, 277.8538, 516.0192],
    #                                      [341.6522, 333.3504, 451.9978, 426.8416],
    #                                      [198.9376, 334.4896, 294.5344, 415.0400],
    #                                      [200.4664, 427.7888, 220.5231, 470.7584],
    #                                      [338.9312, 545.5232, 457.9277, 605.9776],
    #                                      [477.6312, 426.7520, 495.9073, 461.8624],
    #                                      [320.8401, 427.9936, 332.8674, 457.9584],
    #                                      [582.4841, 426.7520, 607.3450, 438.5536]], device='cpu',
    #                                     format=BoundingBoxFormat.XYXY, canvas_size=(640, 640)),
    #              'labels': tensor([17, 17, 0, 0, 0, 58, 0, 0, 0], device='cpu'),
    #              'image_id': tensor([3], device='cpu'),
    #              'area': tensor([24051.4844, 13242.9043, 4798.0059, 3581.3445, 400.8260, 3345.7776,
    #                              298.4381, 167.6160, 136.4558], device='cpu'),
    #              'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], device='cpu'),
    #              'orig_size': tensor([381, 500], device='cpu'), 'idx': tensor([2], device='cpu')}]
