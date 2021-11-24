import argparse
import cv2
import numpy as np

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
from paddle.io import DataLoader

import scipy.misc
import sys
import os
import os.path as osp
import pickle
from packaging import version

from model.deeplab import Res_Deeplab
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2D, BCEWithLogitsLoss2D
from utils.SubsetRandomSampler import SubsetRandomSampler
from dataset.voc_dataset import VOCDataSet, VOCGTDataSet

import matplotlib.pyplot as plt
import random
import timeit
import logging
from tqdm import tqdm

start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 10
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './dataset/VOC2012'
DATA_LIST_PATH = './dataset/voc_list/train_aug.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 20000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = r'./pdparams/resnet101COCO-41f33a49.pdparams'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_ADV_PRED = 0.1

PARTIAL_DATA = 0.5

SEMI_START = 5000
LAMBDA_SEMI = 0.1
MASK_T = 0.2

LAMBDA_SEMI_ADV = 0.001
SEMI_START_ADV = 0
D_REMAIN = True


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to the all data.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--partial-data", type=float, default=PARTIAL_DATA,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--partial-id", type=str, default=None,
                        help="restore partial id list")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-pred", type=float, default=LAMBDA_ADV_PRED,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-semi", type=float, default=LAMBDA_SEMI,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--lambda-semi-adv", type=float, default=LAMBDA_SEMI_ADV,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--mask-T", type=float, default=MASK_T,
                        help="mask T for semi adversarial training.")
    parser.add_argument("--semi-start", type=int, default=SEMI_START,
                        help="start semi learning after # iterations")
    parser.add_argument("--semi-start-adv", type=int, default=SEMI_START_ADV,
                        help="start semi learning after # iterations")
    parser.add_argument("--D-remain", type=bool, default=D_REMAIN,
                        help="Whether to train D with unlabeled data")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument('--debug', action='store_true', help='only do 100 epoch ')
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    # label = Variable(label.long()).cuda(gpu)
    # criterion = CrossEntropy2d().cuda(gpu)
    label = label.astype(paddle.int64)
    criterion = CrossEntropy2D()

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    # optimizer._param_groups[0]['lr'] = lr
    # if len(optimizer._param_groups) > 1:
    #     optimizer._param_groups[1]['lr'] = lr * 10
    optimizer.set_lr(lr)


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    # optimizer._param_groups[0]['lr'] = lr
    # if len(optimizer._param_groups) > 1:
    #     optimizer._param_groups[1]['lr'] = lr * 10
    optimizer.set_lr(lr)


def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:, i, ...] = (label == i)
    # handle ignore labels
    return paddle.to_tensor(one_hot, dtype=paddle.float32)


def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape) * label
    D_label[ignore_mask] = 255
    # D_label = Variable(torch.FloatTensor(D_label)).cuda(args.gpu)
    D_label = paddle.to_tensor(D_label, dtype=paddle.float32)

    return D_label


def main():
    if args.data_path:
        args.data_dir = args.data_path
        args.data_list = os.path.join(args.data_path, 'voc_list/train_aug.txt')

    if args.debug:
        args.num_steps=100
        args.semi_start=10
        args.save_pred_every=10


    logging.basicConfig(
        filename=os.path.join(args.snapshot_dir, 'train_log.txt'),
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger=logging.getLogger()
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    # cudnn.enabled = True
    gpu = args.gpu

    # create network
    model = Res_Deeplab(num_classes=args.num_classes)

    # load pretrained parameters
    saved_state_dict = paddle.load(args.restore_from)

    # only copy the params that exist in current model (caffe-like)
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        print(name)
        if name in saved_state_dict and param.shape == saved_state_dict[name].shape:
            new_params[name].set_value(saved_state_dict[name])
            print('copy {}'.format(name))
    model.set_state_dict(new_params)

    model.train()

    # model.cuda(args.gpu)
    # cudnn.benchmark = True

    # init D
    model_D = FCDiscriminator(num_classes=args.num_classes)
    if args.restore_from_D is not None:
        model_D.set_state_dict(paddle.load(args.restore_from_D))
    model_D.train()
    # model_D.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
                               scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    train_dataset_size = len(train_dataset)

    train_gt_dataset = VOCGTDataSet(args.data_dir, args.data_list, crop_size=input_size,
                                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    if args.partial_data is None:
        trainloader = DataLoader(train_dataset,
                                 batch_size=args.batch_size, shuffle=True, num_workers=5)

        trainloader_gt = DataLoader(train_gt_dataset,
                                    batch_size=args.batch_size, shuffle=True, num_workers=5)
    else:
        # sample partial data
        partial_size = int(args.partial_data * train_dataset_size)

        if args.partial_id is not None:
            train_ids = pickle.load(open(args.partial_id))
            print('loading train ids from {}'.format(args.partial_id))
            logger.info('loading train ids from {}'.format(args.partial_id))
        else:
            train_ids = np.arange(train_dataset_size)
            np.random.shuffle(train_ids)

        pickle.dump(train_ids, open(osp.join(args.snapshot_dir, 'train_id.pkl'), 'wb'))

        train_sampler = SubsetRandomSampler(train_ids[:partial_size])
        train_remain_sampler = SubsetRandomSampler(train_ids[partial_size:])
        train_gt_sampler = SubsetRandomSampler(train_ids[:partial_size])

        train_sampler = paddle.io.BatchSampler(sampler=train_sampler, batch_size=args.batch_size)
        train_remain_sampler = paddle.io.BatchSampler(sampler=train_remain_sampler, batch_size=args.batch_size)
        train_gt_sampler = paddle.io.BatchSampler(sampler=train_gt_sampler, batch_size=args.batch_size)

        trainloader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=3)
        trainloader_remain = DataLoader(dataset=train_dataset, batch_sampler=train_remain_sampler, num_workers=3)
        trainloader_gt = DataLoader(dataset=train_gt_dataset, batch_sampler=train_gt_sampler, num_workers=3)

        trainloader_remain_iter = enumerate(trainloader_remain)

    trainloader_iter = enumerate(trainloader)
    trainloader_gt_iter = enumerate(trainloader_gt)

    # implement model.optim_parameters(args) to handle different models' lr setting

    # optimizer for segmentation network
    optimizer = optim.Momentum(learning_rate=args.learning_rate, parameters=model.parameters(),
                               momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.clear_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(learning_rate=args.learning_rate_D, parameters=model_D.parameters(), beta1=0.9, beta2=0.99)
    optimizer_D.clear_grad()

    # loss/ bilinear upsampling
    bce_loss = BCEWithLogitsLoss2D()
    # interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    pred_label = 0
    gt_label = 1
    with tqdm(range(args.num_steps)) as t:
        for i_iter in range(args.num_steps):
            t.set_description('Train : ')
            loss_seg_value = 0
            loss_adv_pred_value = 0
            loss_D_value = 0
            loss_semi_value = 0
            loss_semi_adv_value = 0

            optimizer.clear_grad()
            adjust_learning_rate(optimizer, i_iter)
            optimizer_D.clear_grad()
            adjust_learning_rate_D(optimizer_D, i_iter)

            for sub_i in range(args.iter_size):

                # train G

                # don't accumulate grads in D
                for param in model_D.parameters():
                    param.stop_gradient = True

                # do semi first
                if (args.lambda_semi > 0 or args.lambda_semi_adv > 0) and i_iter >= args.semi_start_adv:
                    try:
                        _, batch = trainloader_remain_iter.__next__()
                    except:
                        trainloader_remain_iter = enumerate(trainloader_remain)
                        _, batch = trainloader_remain_iter.__next__()

                    # only access to img
                    images, _, _, _ = batch
                    # images = Variable(images).cuda(args.gpu)

                    pred = interp(model(images))
                    pred_remain = pred.detach()

                    D_out = interp(model_D(F.softmax(pred)))
                    D_out_sigmoid = F.sigmoid(D_out).numpy().squeeze(axis=1)

                    ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(bool)

                    loss_semi_adv = args.lambda_semi_adv * bce_loss(D_out, make_D_label(gt_label, ignore_mask_remain))
                    loss_semi_adv = loss_semi_adv / args.iter_size

                    # loss_semi_adv.backward()
                    loss_semi_adv_value += loss_semi_adv.numpy()[0] / args.lambda_semi_adv

                    if args.lambda_semi <= 0 or i_iter < args.semi_start:
                        loss_semi_adv.backward()
                        loss_semi_value = 0
                    else:
                        # produce ignore mask
                        semi_ignore_mask = D_out_sigmoid < args.mask_T

                        semi_gt = pred.numpy().argmax(axis=1)
                        semi_gt[semi_ignore_mask] = 255
                        print(type(semi_ignore_mask))
                        print(semi_ignore_mask)
                        # raise NotImplementedError
                        semi_ratio = 1.0 - sum(semi_ignore_mask) / semi_ignore_mask.size
                        print('semi ratio: {:.4f}'.format(semi_ratio))
                        logger.info('semi ratio: {:.4f}'.format(semi_ratio))

                        if semi_ratio == 0.0:
                            loss_semi_value += 0
                        else:
                            semi_gt = paddle.to_tensor(semi_gt, dtype=paddle.float32)

                            loss_semi = args.lambda_semi * loss_calc(pred, semi_gt, args.gpu)
                            loss_semi = loss_semi / args.iter_size
                            loss_semi_value += loss_semi.numpy()[0] / args.lambda_semi
                            loss_semi += loss_semi_adv
                            loss_semi.backward()

                else:
                    loss_semi = None
                    loss_semi_adv = None

                # train with source

                try:
                    _, batch = trainloader_iter.__next__()
                except:
                    trainloader_iter = enumerate(trainloader)
                    _, batch = trainloader_iter.__next__()

                images, labels, _, _ = batch
                # images = Variable(images).cuda(args.gpu)
                ignore_mask = (labels.numpy() == 255)
                pred = interp(model(images))

                loss_seg = loss_calc(pred, labels, args.gpu)

                D_out = interp(model_D(F.softmax(pred)))

                loss_adv_pred = bce_loss(D_out, make_D_label(gt_label, ignore_mask))

                loss = loss_seg + args.lambda_adv_pred * loss_adv_pred

                # proper normalization
                loss = loss / args.iter_size
                loss.backward()
                loss_seg_value += loss_seg.numpy()[0] / args.iter_size
                loss_adv_pred_value += loss_adv_pred.numpy()[0] / args.iter_size

                # train D

                # bring back stop_gradient
                for param in model_D.parameters():
                    param.stop_gradient = True

                # train with pred
                pred = pred.detach()

                if args.D_remain:
                    pred = paddle.concat((pred, pred_remain), 0)
                    ignore_mask = np.concatenate((ignore_mask, ignore_mask_remain), axis=0)

                D_out = interp(model_D(F.softmax(pred)))
                loss_D = bce_loss(D_out, make_D_label(pred_label, ignore_mask))
                loss_D = loss_D / args.iter_size / 2
                loss_D.backward()
                loss_D_value += loss_D.numpy()[0]

                # train with gt
                # get gt labels
                try:
                    _, batch = trainloader_gt_iter.__next__()
                except:
                    trainloader_gt_iter = enumerate(trainloader_gt)
                    _, batch = trainloader_gt_iter.__next__()

                _, labels_gt, _, _ = batch
                # D_gt_v = Variable(one_hot(labels_gt)).cuda(args.gpu)
                D_gt_v = one_hot(labels_gt)
                ignore_mask_gt = (labels_gt.numpy() == 255)

                D_out = interp(model_D(D_gt_v))
                loss_D = bce_loss(D_out, make_D_label(gt_label, ignore_mask_gt))
                loss_D = loss_D / args.iter_size / 2
                loss_D.backward()
                loss_D_value += loss_D.numpy()[0]

            optimizer.step()
            optimizer_D.step()

            # print('exp = {}'.format(args.snapshot_dir))
            # print(
            #     'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.3f}, loss_semi = {5:.3f}, loss_semi_adv = {6:.3f}'.format(
            #         i_iter, args.num_steps, loss_seg_value, loss_adv_pred_value, loss_D_value, loss_semi_value,
            #         loss_semi_adv_value))
            logger.info('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.3f}, loss_semi = {5:.3f}, loss_semi_adv = {6:.3f}'.format(
                    i_iter, args.num_steps, loss_seg_value, loss_adv_pred_value, loss_D_value, loss_semi_value,loss_semi_adv_value))
            t.set_postfix(iter = f'{i_iter}/{args.num_steps}', loss_seg = f'{loss_seg_value:.3f}', loss_adv_p = f'{loss_adv_pred_value:.3f}', loss_D = f'{loss_D_value:.3f}', loss_semi = f'{loss_semi_value:.3f}', loss_semi_adv = f'{loss_semi_adv_value:.3f}')
            if i_iter >= args.num_steps - 1:
                print('save model ...')
                paddle.save(model.state_dict(), osp.join(args.snapshot_dir, 'VOC_' + str(args.num_steps) + '.pdparams'))
                paddle.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'VOC_' + str(args.num_steps) + '_D.pdparams'))
                break

            if i_iter % args.save_pred_every == 0 and i_iter != 0:
                print('taking snapshot ...')
                paddle.save(model.state_dict(), osp.join(args.snapshot_dir, 'VOC_' + str(i_iter) + '.pdparams'))
                paddle.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'VOC_' + str(i_iter) + '_D.pdparams'))
            t.update()

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
