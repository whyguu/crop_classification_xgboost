import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
# create_feature_map(train_data.columns)


# read in data
if True:
    # train data
    train_data_path = './天池-天体光谱智能分类/data_train_sk/train_data_mean_std.npy'
    train_labels_path = './天池-天体光谱智能分类/train_labels.npy'
    train_data = np.load(train_data_path).astype(np.float32)
    train_labels = np.load(train_labels_path)
    dtrain = xgb.DMatrix(data=train_data, label=train_labels)

    # val data
    val_data = np.load('./天池-天体光谱智能分类/data_val/val_data_mean_std.npy')
    val_labels = np.load('./天池-天体光谱智能分类/data_val/val_labels.npy')
    dval = xgb.DMatrix(data=val_data, label=val_labels)

    # test data
    test_data = np.load('./天池-天体光谱智能分类/data_test/test_data_mean_std.npy')
    dtest = xgb.DMatrix(data=test_data)

    print('data loading done !')

# specify parameters via map
params = {
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
    'num_class': 4,
    'eval_metric': ['merror', 'mlogloss'],  # the last one will be used for early_stopping
    'seed': 0,
}

# model train
if False:
    num_round = 1000
    callbacks = []  # [xgb.callback.reset_learning_rate(custom_rates)]
    eval_list = [(dtrain, 'train'), (dval, 'eval')]
    bst_base = xgb.Booster({'nthread': 32}).load_model('xgb-train-sk-fs11-2.model')  # load data
    # bst_base = None
    bst = xgb.train(params, dtrain, num_round, evals=eval_list, xgb_model=bst_base, early_stopping_rounds=10, callbacks=callbacks,)

    # persistence
    bst.save_model('xgb-train-sk-fs11-3.model')

# model val and test
if False:
    # ##### load model #####
    bst = xgb.Booster({'nthread': 32}).load_model('xgb-train-sk-fs11-3.model')  # init model & load data

    # # val metric
    val_prob = bst.predict(dval)
    val_prob = np.reshape(val_prob, (-1, 4))
    val_rlt = np.argmax(val_prob, axis=1)
    acc = accuracy_score(y_true=val_labels, y_pred=val_prob)
    f1 = f1_score(y_true=val_labels, y_pred=val_rlt, average='macro')
    report = classification_report(y_true=val_labels, y_pred=val_rlt, target_names=['star', 'galaxy', 'qso', 'unknown'])
    print('accuracy: ', acc)
    print('f1_score: ', f1)
    print('classification_report: \n', report)

    # test
    test_preds = bst.predict(dtest)
    test_preds = np.reshape(test_preds, (-1, 4))
    test_preds = np.argmax(test_preds, axis=1)

    test_index_path = './天池-天体光谱智能分类/index/first_test_index_20180131.csv'
    test_index = pd.read_csv(test_index_path)
    test_index = pd.concat([test_index, pd.DataFrame(data=test_preds, columns=['type'])], axis=1)
    type_name = ['star', 'galaxy', 'qso', 'unknown']
    for idx, ty_name in enumerate(type_name):
        print(idx, ty_name)
        test_index.replace(to_replace=idx, value=ty_name, inplace=True)
    test_index.to_csv('./astronomy-xgb.csv', index=None, header=None)

# model analysis
if True:
    bst = xgb.Booster({'nthread': 32}).load_model('/Users/whyguu/Desktop/xgbst-sk-fs11.model')  # load data
    # attributes
    attr = bst.attributes()
    print(attr)
    # dump model
    dump = bst.get_dump()
    print(type(dump))
    print(len(dump))
    print(dump[0])
    bst.dump_model('tree_dump.txt')
    # feature importance
    plot_importance(booster=bst, max_num_features=30, height=0.1)
    plt.savefig('xgb_feature_importance.png', dpi=100)
    # plot tree
    plt.figure()
    plot_tree(booster=bst, fmap='', num_trees=0)
    fig = plt.gcf()
    fig.set_size_inches(100, 50)
    fig.savefig('tree.png')



