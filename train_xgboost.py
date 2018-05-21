# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import os
import sys
import argparse
import tifffile as tiff
import gc
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from skimage import io
from tools import read_img, write_img, arg_parser, read_tiff


def make_data(image, label, class_idx, balance=False):
    crops = []
    for idx in class_idx:
        tp = np.reshape(image[label == idx, :], newshape=(-1, image.shape[-1]))
        crops.append(tp)
    data_num = [crop.shape[0] for crop in crops]
    print(data_num)
    # # balance data
    if balance:
        tp_num = np.min(data_num)
        data_num = MAX_TRAIN_DATA_NUM if tp_num > MAX_TRAIN_DATA_NUM else tp_num
        feature_num = image.shape[-1]
        train_data = np.zeros(shape=(data_num * len(crops), feature_num), dtype=image.dtype)
        train_labels = np.zeros(shape=data_num * len(crops))
        for i, crop in enumerate(crops):
            # you can shufful crop here before make train_data
            # Todo
            train_data[i * data_num:(i + 1) * data_num, :] = crop[0:data_num, :]
            train_labels[i * data_num:(i + 1) * data_num] = i  # CROP_NAMES_AND_LABEL_INDEX[i][1]
    else:
        train_data = np.concatenate(tuple(crops), axis=0)
        train_labels = np.zeros(train_data.shape[0])
        for i in range(1, len(crops)):
            train_labels[np.sum(data_num[0:i]): np.sum(data_num[0:i+1])] = i
    for i in range(len(crops)):
        assert np.sum(train_labels == i) == crops[i].shape[0]

    crops = None
    gc.collect()
    # shuffle
    for i in range(3):
        train_data, train_labels = utils.shuffle(train_data, train_labels, )
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels,
                                                                      test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(data=train_data, label=train_labels)
    dval = xgb.DMatrix(data=val_data, label=val_labels)

    return dtrain, dval


def xgb_train(dtrain, dval, class_num, print_val=True, base_model_path=None):

    # base model
    if base_model_path is not None:
        bst_base = xgb.Booster({'nthread': 32})  # init model
        bst_base.load_model(base_model_path)  # load data
    else:
        bst_base = None

    callbacks = []  # [xgb.callback.reset_learning_rate(custom_rates)]
    eval_list = [(dtrain, 'train'), (dval, 'eval')]

    # train
    XGB_PARAMS['num_class'] = class_num
    bst = xgb.train(XGB_PARAMS, dtrain, NUM_ROUND, evals=eval_list, xgb_model=bst_base,
                    early_stopping_rounds=10, callbacks=callbacks, )

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


def check_before_run():
    # Todo
    pass


# global variables
# CROP_NAMES_AND_LABEL_INDEX = [('corn', 1), ('bean', 2), ('peanut', 3), ('other', 4)]

MAX_TRAIN_DATA_NUM = 10000000

NUM_ROUND = 1000

XGB_PARAMS = {
    # General Parameters
    'booster': 'gbtree',
    'silent': 1,
    'nthread': 32,
    # Parameters for Tree Booster
    'eta': 0.2,  # learning_rate
    'gamma': 0,  # [default=0, alias: min_split_loss]
    'min_child_weight': 1,
    'max_depth': 4,  # [default=0, alias: min_split_loss]
    'subsample': 0.2,  # [default=1] 每次采样 data 的 subsample 来进行树增长
    'colsample_bytree': 0.2,  # 每次采样 feature 的 subsample 来进行树增长
    'lambda': 1,  # L2 regularization
    'alpha': 0,  # L1 regularization
    'tree_method': 'gpu_exact',  # [default='auto']
    'gpu_id': 0,
    'n_gpus': 1,  # only valid for tree_method 'gpu_hist'
    'process_type': 'default',  # [default='default']
    'grow_policy': 'depthwise',  # string [default='depthwise'] Choices: {'depthwise', 'lossguide'}

    'max_leaves': 0,  # [default=0] Only relevant for the ‘lossguide’ grow policy.
    'max_bin': 256,  # [default=256] only used if ‘hist’ is specified as tree_method

    'predictor': 'gpu_predictor',  # [default='cpu_predictor']
    # attention: there are Additional parameters for Dart Booster

    # Learning Task Parameters
    'objective': 'multi:softprob',  # the output of multi:softprob is a vector of ndata * nclass,
    # which can be further reshaped to ndata, nclass matrix.
    # 'num_class': len(class_idx),
    'eval_metric': ['merror', 'mlogloss'],  # the last one will be used for early_stopping
    'seed': 0,
}


if __name__ == '__main__':
    flags = arg_parser()

    # read image/label (height width channels)
    image = read_tiff(flags.image)
    label = read_tiff(flags.label)

    zero_mask = image[:, :, 0] == 0

    class_idx = np.unique(label)
    class_idx = sorted([class_idx[i] for i in range(class_idx.shape[0]) if class_idx[i] != 0])

    print(np.unique(label))
    print('image: ', image.shape)
    print('label: ', label.shape)

    # memory check and so on
    check_before_run()

    # make train val data
    dtrain, dval = make_data(image, label, class_idx)
    gc.collect()
    # xgb train
    bst = xgb_train(dtrain=dtrain, dval=dval, class_num=len(class_idx))
    # # persistence
    bst.save_model('./xgb-train.model')

    # predict & output
    prediction = bst.predict(xgb.DMatrix(np.reshape(image, newshape=(-1, image.shape[-1]))))
    prediction = np.reshape(prediction, (-1, len(class_idx)))
    prediction = np.reshape(np.argmax(prediction, axis=1), newshape=image.shape[0:2]).astype(np.uint8)
    print(np.unique(np.ravel(prediction)))

    # make predicted map and label map have the same index
    ind_mask = []
    for i, num in enumerate(class_idx):
        ind_mask.append(prediction == i)
        # print(i, np.sum(ind_mask[-1]))
    for i, num in enumerate(class_idx):
        prediction[ind_mask[i]] = num
    prediction[zero_mask] = 0

    print(np.unique(np.ravel(prediction)))

    # generate pseudo color
    colors = np.random.randint(0, 255, size=(len(class_idx)*3, ), dtype=np.uint8).reshape(-1, 3)
    pseudo = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    print(colors)
    for i, idx in enumerate(class_idx):
        pseudo[prediction == idx, :] = colors[i, :]

    im_proj, im_geotrans = read_img(flags.image, only_coordinate=True)
    write_img(filename=flags.output, im_proj=im_proj, im_geotrans=im_geotrans, im_data=prediction)
    # io.imsave(flags.output, prediction)
    io.imsave('./pseudo.png', pseudo)











