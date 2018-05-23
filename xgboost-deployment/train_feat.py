# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import warnings
import os
import gc
from scipy.ndimage import filters
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from skimage import io
from tools import *
from sklearn.metrics import confusion_matrix
import time
import argparse


# global variables
MAX_TRAIN_DATA_NUM = 10000000

NUM_ROUND = 1000

USE_GPU = True
GPU_ID = 0
GPU_NUM = 1  # gpu num greater than 1 , gpu id will be useless

TOP_K = 3

XGB_PARAMS = {
    # General Parameters
    'booster': 'gbtree',
    'silent': 1,
    'nthread': 4,
    # Parameters for Tree Booster
    'eta': 0.2,  # learning_rate
    'gamma': 0,  # [default=0, alias: min_split_loss]
    'min_child_weight': 1,
    'max_depth': 4,  # [default=0, alias: min_split_loss]
    'subsample': 0.2,  # [default=1] 每次采样 data 的 subsample 来进行树增长
    'colsample_bytree': 0.2,  # 每次采样 feature 的 subsample 来进行树增长
    'lambda': 1,  # L2 regularization
    'alpha': 0,  # L1 regularization
    # about gpus
    'tree_method': 'gpu_exact',  # [default='auto'] gpu_exact
    'max_bin': 256,  # [default=256] only used if ‘hist’ is specified as tree_method
    'gpu_id': 0,
    'n_gpus': 1,  # only valid for tree_method 'gpu_hist'

    'process_type': 'default',  # [default='default']
    'grow_policy': 'depthwise',  # string [default='depthwise'] Choices: {'depthwise', 'lossguide'}

    'max_leaves': 0,  # [default=0] Only relevant for the ‘lossguide’ grow policy.

    'predictor': 'gpu_predictor',  # [default='cpu_predictor']
    # attention: there are Additional parameters for Dart Booster

    # Learning Task Parameters
    'objective': 'multi:softprob',  # the output of multi:softprob is a vector of ndata * nclass,
    # which can be further reshaped to ndata, nclass matrix.
    # 'num_class': len(class_idx),
    'eval_metric': ['merror', 'mlogloss'],  # the last one will be used for early_stopping
    'seed': 0,
}

WIN_SIZE = 5

GPU_MEMORY = 7.5  # GB


def generate_feature_for_predict(img, bd_list, win_size=WIN_SIZE):
    channel_feat = ['mean', 'std/var', 'range', ]  # 'mutual_info', 'entropy']
    img = img[:, :, bd_list]
    band_num = len(bd_list)

    if win_size > 0:
        feat_num = band_num * (band_num - 1) // 2 + len(channel_feat) * band_num
    else:
        feat_num = band_num * (band_num - 1) // 2
    new_feat_data = np.zeros(shape=(img.shape[0], img.shape[1], feat_num), dtype=np.float32)

    pointer = 0
    for i in range(band_num - 1):
        toke = band_num - i - 1
        new_feat_data[:, :, pointer:pointer + toke] = np.expand_dims(img[:, :, i], axis=-1) / (img[:, :, i + 1:] + 1e-5)
        pointer += toke

    h, w, _ = image.shape
    if win_size > 0:
        # mean
        channel_mean = new_feat_data[:, :, pointer:pointer + band_num] = filters.convolve(input=img, weights=np.ones(shape=(win_size, win_size, 1))/win_size/win_size) # the same as filters.uniform_filter with size=[win_size, win_size, 1]
        # channel_mean = new_feat_data[:, :, pointer:pointer + band_num] = filters.uniform_filter(input=\
        # img.astype(np.float32), size=[win_size, win_size, 0], )  # dtype of output is the same as input
        pointer += band_num

        # std
        channel_square_mean = filters.convolve(input=img*img, weights=np.ones(shape=(win_size, win_size, 1))/win_size/win_size)
        new_feat_data[:, :, pointer:pointer + band_num] = np.sqrt(channel_square_mean - channel_mean*channel_mean)
        pointer += band_num
        channel_square_mean = None

        # range
        new_feat_data[:, :, pointer:pointer + band_num] = filters.maximum_filter(img, size=[win_size, win_size, 1]) - \
            filters.minimum_filter(img, size=[win_size, win_size, 1])
        pointer += band_num

        # mutual info
        # Todo

        # entropy
        # Todo
    return new_feat_data


def prepare_data_for_second_train(data, coordinate, feat):
    ft = feat[coordinate[:, 0], coordinate[:, 1], :]
    data = np.concatenate((data, ft.reshape(-1, ft.shape[-1])), axis=-1)

    return data


def first_train(data, labels, top_k, class_num, each_class_data=10000):
    # 1.1 select 1000 data to train
    first_data, first_labels = choose_data(data=data, labels=labels, number=min(data.shape[0], each_class_data*class_num), )
    first_data_train, first_data_val = organize_data_for_xgb_train(data=first_data, labels=first_labels)

    # 1.2 train
    bst = xgb_train(dtrain=first_data_train, dval=first_data_val, class_num=class_num)

    # 1.3 select features by importance
    # bst = xgb.Booster()
    feat_names = ['band' + str(i+1) for i in range(data.shape[-1])]
    path = os.path.dirname(os.path.abspath(__file__))
    create_feature_map(features=feat_names, out_name=path + '/band_select_feature.fmap')
    first_feature_score = bst.get_score(fmap=path + 'band_select_feature.fmap', importance_type='gain')
    first_feature_score = sorted(first_feature_score.items(), key=lambda x: x[1], reverse=True)
    print('sorted band_select_feature_score:')
    for score in first_feature_score:
        print('{}: {:.3f}'.format(score[0], score[1]))
    # 1.4 select feature
    top_k = min(top_k, data.shape[-1])
    first_ft_list = [first_feature_score[i][0] for i in range(top_k)]
    bd_list = [int(first_feature_score[i][0][4:])-1 for i in range(top_k)]
    return first_ft_list, bd_list


def second_train(data, labels, class_num):
    da_train, da_val = organize_data_for_xgb_train(data=data, labels=labels)
    # 1.2 train
    boost = xgb_train(dtrain=da_train, dval=da_val, class_num=class_num)
    return boost


def predict(img, feat, boost, cls_num):
    bg = time.time()
    # new_feat = generate_feature_for_predict(image, band_list)
    dt_pred = np.concatenate((img.reshape(-1, img.shape[-1]).astype(np.float32), feat.reshape(-1, feat.shape[-1])), axis=-1)
    h, w, _ = img.shape
    img = None
    # print('feature generated.')
    # boost = xgb.Booster()
    print('predicting, please wait.')
    if USE_GPU:
        mem_need = dt_pred.nbytes / 1024 / 1204 / 1024

        cmd = "nvidia-smi  -i " + str(GPU_ID) + " -q --display=MEMORY | grep Free | head -1 | awk '{print $(NF-1)}'"
        rlt = os.popen(cmd)
        mem = int(rlt.read()) / 1024.0
        print('Memory Need: {:.3f}'.format(mem_need))
        print('Memory Free: {:.3f}'.format(mem))
        if mem > mem_need + 0.2:
            print('using gpu for predict.')
            pred = boost.predict(xgb.DMatrix(dt_pred))
        else:
            print('using cpu for predict.')
            # XGB_PARAMS['predictor'] = 'cpu_predictor'
            boost.set_param({'predictor': 'cpu_predictor'})
            pred = boost.predict(xgb.DMatrix(dt_pred))
    else:
        pred = boost.predict(xgb.DMatrix(dt_pred))

    pred = np.reshape(pred, (-1, cls_num))
    pred = np.reshape(np.argmax(pred, axis=1), newshape=(h, w)).astype(np.uint8)
    # print('predict done !', (time.time()-bg)/60)
    return pred


def post_process(prediction, class_idx, z_mask, out_path):
    # print('post processing ...')
    # 4. make predicted map and label map have the same index
    # print('prediction', np.unique(np.ravel(prediction)))

    class_idx.reverse()
    for i, num in enumerate(class_idx):
        ind_mask = (prediction == len(class_idx) - i - 1)
        # print(i, np.sum(ind_mask[-1]))
        prediction[ind_mask] = num

    # print('prediction', np.unique(np.ravel(prediction)))
    prediction[z_mask.astype(np.bool)] = 0
    # print('prediction', np.unique(np.ravel(prediction)))

    # 5. store prediction result
    im_projection, im_geo_transformation = read_img(flags.image, only_coordinate=True)
    write_img(filename=out_path, im_proj=im_projection, im_geotrans=im_geo_transformation, im_data=prediction)
    # print('post process done ! now store the image')
    # io.imsave(out_path, prediction)
    class_idx.reverse()
    return prediction


def choose_data(data, labels, number, coordinate=None):
    index = np.random.randint(0, labels.shape[0], number)
    if coordinate is None:
        return data[index, :], labels[index]
    else:
        return data[index, :], labels[index], coordinate[index, :]


def make_data(img, lab, class_indices, coordinate):
    """
    img: image
    lab: label
    class_indices: index of classes
    """
    crops = []
    crops_coordinate = []
    for idx in class_indices:
        img_idx = lab == idx
        crops.append(np.reshape(img[img_idx, :], newshape=(-1, img.shape[-1])))
        crops_coordinate.append(np.reshape(coordinate[img_idx, :], newshape=(-1, coordinate.shape[-1])))

    data_num = [crop.shape[0] for crop in crops]
    # print('train_data nums', data_num)

    train_data = np.concatenate(tuple(crops), axis=0)
    train_data_coordinate = np.concatenate(tuple(crops_coordinate), axis=0)
    train_labels = np.zeros(train_data.shape[0])
    for i in range(1, len(crops)):
        train_labels[np.sum(data_num[0:i]): np.sum(data_num[0:i+1])] = i
    for i in range(len(crops)):
        assert np.sum(train_labels == i) == crops[i].shape[0]

    crops = None
    coordinate = None
    gc.collect()
    # shuffle
    for i in range(3):
        train_data, train_labels, train_data_coordinate = utils.shuffle(train_data, train_labels, train_data_coordinate)
    return train_data, train_labels, train_data_coordinate


# memory check
def check_before_run():
    # Todo
    pass


def xgb_train(dtrain, dval, class_num, print_val=False, base_model_path=None):

    # base model
    if base_model_path is not None:
        bst_base = xgb.Booster({'nthread': 32})  # init model
        bst_base.load_model(base_model_path)  # load data
    else:
        bst_base = None
    XGB_PARAMS['num_class'] = class_num

    callbacks = []  # [xgb.callback.reset_learning_rate(custom_rates)]
    eval_list = [(dtrain, 'train'), (dval, 'eval')]
    # train
    bst = xgb.train(XGB_PARAMS, dtrain, NUM_ROUND, evals=eval_list, xgb_model=bst_base,
                    early_stopping_rounds=10, callbacks=callbacks, verbose_eval=False)

    # # val metric
    if print_val:
        val_preds = bst.predict(dval)
        val_preds = np.reshape(val_preds, (-1, class_num))
        val_preds_rlt = np.argmax(val_preds, axis=1)
        acc = accuracy_score(y_true=dval.get_label(), y_pred=val_preds_rlt)
        f1 = f1_score(y_true=dval.get_label(), y_pred=val_preds_rlt, average='macro')
        report = classification_report(y_true=dval.get_label(), y_pred=val_preds_rlt,)
        print('accuracy: ', acc)
        print('f1_score: ', f1)
        print('classification_report: \n', report)

    return bst


def organize_data_for_xgb_train(data, labels, split_ratio=0.2):
    data0, data1, labels0, labels1 = train_test_split(data, labels, test_size=split_ratio, random_state=42)
    xgb_train_data = xgb.DMatrix(data=data0, label=labels0)
    xgb_val_data = xgb.DMatrix(data=data1, label=labels1)
    return xgb_train_data, xgb_val_data


def create_feature_map(features, out_name=None):
    if out_name is None:
        out_name = 'xgb.fmap'
    outfile = open(out_name, 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def prepare_data(img_path, label_path):
    # read image/label (height width channels)
    image = read_tiff(img_path)
    label = read_tiff(label_path).astype(np.uint8)
    print('image: ', image.shape)
    print('label: ', label.shape)
    print('image: ', image.dtype)
    print('label: ', label.dtype)
    print('image and label has been read.')
    zero_mask = np.sum(image == 0, axis=-1)
    zero_mask = zero_mask != 0
    # print(zero_mask.shape)
    # print('sum zero mask:', np.sum(zero_mask))
    class_idx = np.unique(label)
    # print('class_idx: ', class_idx)
    class_idx = sorted([class_idx[i] for i in range(class_idx.shape[0]) if class_idx[i] != 0])
    print('class_idx: ', class_idx)
    h, w, c = image.shape
    co_r, co_c = np.meshgrid(range(h), range(w), indexing='ij')
    coord = np.concatenate((co_r[:, :, np.newaxis], co_c[:, :, np.newaxis]), axis=-1)

    assert image.shape[0:2] == label.shape
    return image, label, coord, zero_mask, class_idx


def gpu_config():

    if not USE_GPU:
        XGB_PARAMS['tree_method'] = 'exact'  # hist
        XGB_PARAMS['predictor'] = 'gpu_predictor'
        return

    if GPU_NUM == 1:
        XGB_PARAMS['gpu_id'] = GPU_ID
    else:
        XGB_PARAMS['tree_method'] = 'gpu_hist'

    XGB_PARAMS['n_gpus'] = GPU_NUM


# argument
def arg_parser():
    parser = argparse.ArgumentParser()
    path = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument('-image', type=str, required=False, default=path + '/image.tif')
    parser.add_argument('-label', type=str, required=False, default=path + '/label.tif')
    parser.add_argument('-output', type=str, required=False, default=path + '/out.tif')

    parser.add_argument('-top_k', type=int, required=False, default=3)
    parser.add_argument('-win_size', type=int, required=False, default=5)
    parser.add_argument('-use_gpu', type=bool, required=False, default=True)
    parser.add_argument('-gpu_id', type=int, required=False, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    flags = arg_parser()
    TOP_K = flags.top_k
    WIN_SIZE = flags.win_size
    USE_GPU = flags.use_gpu
    GPU_ID = flags.gpu_id
    # gpu configuration
    gpu_config()

    begin_time = time.time()

    t0 = time.time()
    image, label, coord, zero_mask, class_idx = prepare_data(flags.image, flags.label)
    print('read and prepare data {:.3f} minutes'.format((time.time() - t0)/60))
    # memory check and so on
    check_before_run()

    # make train label data
    t0 = time.time()
    data_train, data_labels, data_coordinate = make_data(image, label, class_idx, coord)
    _, data_train, _, data_labels, _, data_coordinate = train_test_split(data_train, data_labels, data_coordinate,
                                                                         test_size=0.01, random_state=42)
    print('make train data {:.3f} minutes'.format((time.time() - t0)/60))
    gc.collect()

    # select top k bands
    t0 = time.time()
    selected_band_names, band_list = first_train(data_train, data_labels, TOP_K, len(class_idx))
    print('select band {:.3f} minutes'.format((time.time() - t0)/60))

    # generate feature
    t0 = time.time()
    new_feat = generate_feature_for_predict(image, band_list)
    print('generate feature of train data {:.3f} minutes'.format((time.time() - t0)/60))

    # train
    t0 = time.time()
    feat_data = prepare_data_for_second_train(data_train, data_coordinate, new_feat)
    bst = second_train(feat_data, data_labels, len(class_idx))
    # bst.save_model('./boost.model')
    print('second train {:.3f} minutes'.format((time.time() - t0)/60))

    # predict
    t0 = time.time()
    prediction = predict(image, new_feat, bst, len(class_idx))
    print('prediction {:.3f} minutes'.format((time.time() - t0) / 60))

    # post_process
    t0 = time.time()
    prediction = post_process(prediction, class_idx, zero_mask, flags.output)
    print('post process {:.3f} minutes'.format((time.time() - t0) / 60))

    print('{:.3f} minutes passed.'.format((time.time()-begin_time)/60))

    # pseudo_map(prediction, class_idx, './pseudo.png')

    # cfmx = confusion_matrix(y_true=label.reshape(-1, 1), y_pred=prediction.reshape(-1, 1), \
    # labels=int(['1', '2', '3', '4', '5', '6', '7', '9', '10', '11']))
    # plot_confusion_matrix(cm, classes=['1', '2', '3', '4', '5', '6', '7', '9', '10', '11'], normalize=True)








