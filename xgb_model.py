import pandas as pd
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split


def xgb_cv(X_train, Y_train):
    cv_params = {
        # 'n_estimators': range(125, 180, 5),
        # 'max_depth': range(7, 12, 1),
        # 'min_child_weight': range(4, 5, 1),
        # 'subsample': [0.95, 0.9],
        # 'colsample_bytree': [0.7, 0.6, 0.5],
        # 'gamma': [0.9, 0.85, 0.95, 0.97]
        # 'reg_alpha': [7, 8, 9],
        # 'reg_lambda': [3, 4, 5],
        # 'learning_rate': [0.05, 0.1, 0.15, 0.01]
    }
    model = xgb.XGBClassifier(
        learning_rate=0.05,
        n_estimators=165,
        max_depth=7,
        min_child_weight=4,
        seed=0,
        subsample=0.9,
        colsample_bytree=0.5,
        gamma=0.95,
        reg_alpha=8,
        reg_lambda=3,
        eval_metric='auc')
    optimized_GBM = GridSearchCV(
        estimator=model,
        param_grid=cv_params,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=4)
    optimized_GBM.fit(X_train, Y_train)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i += 1
    outfile.close()


if __name__ == '__main__':
    train_data = pd.read_csv('../deal_data/train_data_flag.csv')
    train_data.fillna(-999, inplace=True)
    train_drop_list = [
        'USRID', 'FLAG', 'user_var_day_2', 'evt_lbl_2_count',
        'user_median_hour_2', 'evt_llbl02_ratio_7', 'V3_-0.37835',
        'user_std_hour_2', 'V3_-0.02316', 'evt_llbl12_ratio_7',
        'user_var_hour_2', 'V3_-0.55595', 'evt_lbl_0_median_15', 'V3_-0.91115',
        'evt_lbl_0_median_7', 'V3_0.8648299999999999'
    ]
    X_train = train_data.drop(train_drop_list, axis=1)
    Y_train = train_data['FLAG']
    # xgb_cv(X_train, Y_train)

    x_train, x_test, y_train, y_test = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=1017)
    xgb_train = xgb.DMatrix(x_train, y_train)
    # xgb_train = xgb.DMatrix(X_train, Y_train)
    xgb_eval = xgb.DMatrix(x_test, y_test)

    params = {
        'objective': 'binary:logistic',
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 4,
        'seed': 0,
        'subsample': 0.9,
        'colsample_bytree': 0.5,
        'gamma': 0.95,
        'reg_alpha': 8,
        'reg_lambda': 3,
        'eval_metric': 'auc'
    }

    watchlist = [(xgb_eval, 'eval'), (xgb_train, 'train')]

    model = xgb.train(
        params=params,
        dtrain=xgb_train,
        num_boost_round=155,
        early_stopping_rounds=20,
        evals=watchlist)

    features = X_train.columns.tolist()
    ceate_feature_map(features)

    # print(model.get_fscore(fmap='xgb.fmap'))

    # importance_dict = sorted(
    #     model.get_fscore(fmap='xgb.fmap').items(),
    #     key=lambda x: x[1],
    #     reverse=True)
    # print(importance_dict)

    # predictors = [i for i in X_train.columns]
    # feat_imp = pd.Series(
    #     model.get_fscore(fmap='xgb.fmap'),
    #     predictors).sort_values(ascending=False)
    # print(feat_imp)

    dropList = []
    for key, val in model.get_fscore(fmap='xgb.fmap').items():
        if val <= 10:
            dropList.append(key)
    print(dropList)

    print('训练集的特征数目为：', len(X_train.columns))

    result = model.predict(xgb.DMatrix(x_test))
    print('AUC值为：', metrics.roc_auc_score(y_test, result))

    test_data = pd.read_csv('../deal_data/test_data.csv')
    test_data.fillna(-999, inplace=True)
    test_drop_list = [
        'USRID', 'user_var_day_2', 'evt_lbl_2_count', 'user_median_hour_2',
        'evt_llbl02_ratio_7', 'V3_-0.37835', 'user_std_hour_2', 'V3_-0.02316',
        'evt_llbl12_ratio_7', 'user_var_hour_2', 'V3_-0.55595',
        'evt_lbl_0_median_15', 'V3_-0.91115', 'evt_lbl_0_median_7',
        'V3_0.8648299999999999'
    ]
    X_test = test_data.drop(test_drop_list, axis=1)
    print('测试集的特征数目为：', len(X_test.columns))
    result = model.predict(xgb.DMatrix(X_test))
    print(result)
    submit = pd.DataFrame()
    submit['USRID'] = test_data['USRID']
    submit['FLAG'] = result
    submit.to_csv('submit_result0705.csv', index=None, sep='\t')