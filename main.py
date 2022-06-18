import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from scipy.stats import norm
path = "D:\机器学习课设/"

train_df = pd.read_csv(path+"candidate_train.csv", encoding='utf-8')
train_answer = pd.read_csv(path+"train_answer.csv", encoding='utf-8')

print(train_answer.head())
print(train_answer.corr(method='pearson', min_periods=1))

sns.heatmap(train_answer.corr(method='pearson', min_periods=1),annot=True, vmax=1,vmin = 0, cmap="YlGnBu")
plt.show()

print(train_df.head())
print(train_df.shape)

print(train_df["0"].nunique())

cols = []
train_nuniq = []
uniq_num = []
for i in train_df.columns:
    cols.append(i)
    train_nuniq.append(train_df[i].nunique())
    uniq_num.append(sorted(train_df[i].value_counts().to_list())[0]) #特征中最少类别的个数
feat_drop = pd.DataFrame({'columns':cols,
                                   'train_nuniq':train_nuniq,
                                   "uniq_num":uniq_num} ).sort_values(by=["train_nuniq","uniq_num"],ascending=True)
feat_drop.to_csv('D:\机器学习课设/columns_nunique.csv',index=None,encoding='utf-8-sig')
feat_drop = pd.read_csv('D:\机器学习课设/columns_nunique.csv', encoding='utf-8')
tmp = feat_drop.groupby("train_nuniq", as_index=False)['columns'].agg({"count":"count"})

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))
autolabel(plt.bar(x=tmp.index, height=tmp['count'], color='r',tick_label=tmp['train_nuniq']))
plt.show()

drop_col = feat_drop[feat_drop["train_nuniq"]<=2]["columns"].to_list()
feats_df = train_df.drop(drop_col, axis=1)
#偏度和峰度图，可以做异常值处理用
sns.distplot(feats_df["3170"], fit=norm)
(mu, sigma) = norm.fit(feats_df["3170"])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('Distribution')
plt.show()

#特征以及特征和target之间的相关性
answer2df = pd.merge(train_answer, feats_df, on=["id"], how='left')
sns.heatmap(answer2df.corr(method='pearson', min_periods=1),annot=True, vmax=1,vmin = 0, cmap="YlGnBu")
plt.show()

print(train_df["7"].value_counts())
print(train_df["7"].value_counts()/train_df.shape[0])

# baseline,成绩10.5
# coding=utf-8
import os
import time
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import KFold

path = "D:\机器学习课设/"


def data_set():
    """特征的nunique，统计，作为筛选特征的标准
    """
    fpaths = path + "molecule_open_data/"
    if not os.path.exists('D:\机器学习课设/columns_nunique.csv'):
        df = pd.read_csv(fpaths + "candidate_train.csv", encoding='utf-8')
        df2 = pd.read_csv(fpaths + "candidate_val.csv", encoding='utf-8')
        cols = [ ]
        train_nuniq = [ ]
        test_nuniq = [ ]
        uniq_num = [ ]
        for i in tqdm(df.columns):
            cols.append(i)
            train_nuniq.append(df[ i ].nunique())
            test_nuniq.append(df2[ i ].nunique())
            uniq_num.append(sorted(df[ i ].value_counts().to_list())[ 0 ])  # 特征中个数最少的类别的个数
        feat_drop = pd.DataFrame({'columns': cols,
                                  'train_nuniq': train_nuniq,  # 特征的类别数目
                                  "test_nuniq": test_nuniq,
                                  "uniq_num": uniq_num  # 特征类别中数量最少类别的数量
                                  }).sort_values(by=[ "train_nuniq", "uniq_num" ], ascending=True)
        feat_drop.to_csv('D:\机器学习课设/columns_nunique.csv', index=None, encoding='utf-8-sig')
    feat_drop = pd.read_csv('D:\机器学习课设/columns_nunique.csv', encoding='utf-8')
    feat_drop = feat_drop[
        (feat_drop[ "train_nuniq" ] == 1) | ((feat_drop[ "train_nuniq" ] == 2) & (feat_drop[ "uniq_num" ] <= 10000)) ]
    # 这里是删除了部分nunique<=2的特征
    #     feat_drop = feat_drop[feat_drop["train_nuniq"]<=2]
    drop_col = feat_drop[ "columns" ].to_list()
    train_answer = pd.read_csv(fpaths + "train_answer.csv", encoding='utf-8')
    df = pd.read_csv(fpaths + "candidate_train.csv", encoding='utf-8')
    df.drop(drop_col, axis=1, inplace=True)
    df = pd.merge(train_answer, df, on=[ "id" ], how='left')
    test_data = pd.read_csv(fpaths + "candidate_val.csv", encoding='utf-8')
    test_data.drop(drop_col, axis=1, inplace=True)
    print(test_data.shape)
    return df, test_data


class Train_model():

    def __init__(self):
        self.target = [ "p1", "p2", "p3", "p4", "p5", "p6" ]

    def data(self):
        self.train_data, self.test_data = data_set()

    def train(self, target, params):
        """模型训练
  
        """
        train_data = self.train_data.copy()
        test_data = self.test_data.copy()
        rows = [ "id", "p1", "p2", "p3", "p4", "p5", "p6" ]
        train_label = train_data[ [ target ] ].copy()
        sub = test_data[ [ "id" ] ].copy()
        train_data.drop(rows, axis=1, inplace=True)
        test_data.drop([ "id" ], axis=1, inplace=True)
        kf = KFold(n_splits=5, shuffle=True, random_state=520)
        kfs = kf.split(train_data, train_label)
        res = [ ]
        ss = 0
        for i, (train_index, vaild_index) in enumerate(kfs):
            print('target:{}-第{}次训练...'.format(target, i + 1))
            train_x = train_data.iloc[ train_index ]
            train_y = train_label.iloc[ train_index ]
            vaild_x = train_data.iloc[ vaild_index ]
            vaild_y = train_label.iloc[ vaild_index ]
            lgb_train = lgb.Dataset(train_x, train_y, silent=True)
            lgb_eval = lgb.Dataset(vaild_x, vaild_y, reference=lgb_train, silent=True)
            print("Memory free: {:2.4f} GB".format(psutil.virtual_memory().free / (1024 ** 3)))
            gbm = lgb.train(params, lgb_train, num_boost_round=10000, valid_sets=[ lgb_train, lgb_eval ],
                            verbose_eval=500, early_stopping_rounds=500)
            test_pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)
            vaild_pred = gbm.predict(vaild_x, num_iteration=gbm.best_iteration)
            res.append(test_pred)
            score = self.smape(vaild_y, vaild_pred)
            ss += score
            # 根据不同的target的特征重要性，可以分别做特征工程
            importance = gbm.feature_importance()
            feature_name = gbm.feature_name()
            feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance}).sort_values(
                by='importance', ascending=False)
            feature_importance.to_csv('D:\机器学习课设/feature_importance_{}_{:2.5f}.csv'.format(target, score), index=None,
                                      encoding='utf-8-sig')
            print('target:{}-第{}次训练-smape:{:2.5f} \n'.format(target, i + 1, score))
            print("\n")
        print('target:{}-五折训练分数均值-smape:{:2.5f} \n'.format(target, ss / (i + 1)))
        res = np.array(res)
        res = res.mean(axis=0)
        sub[ target ] = res
        sub.to_csv('sub_{}.csv'.format(target), index=None, encoding='utf-8')

    def sub_result(self):
        """生成的提交文件
        """
        sub = pd.read_csv('sub_{}.csv'.format(self.target[ 0 ]), encoding='utf-8')
        for target in self.target[ 1: ]:
            sub1 = pd.read_csv('sub_{}.csv'.format(target), encoding='utf-8')
            sub = pd.merge(sub, sub1, on=[ "id" ], how='left')
        sub.to_csv('submission.csv', index=None, encoding='utf-8')

    def smape(self, y_true, y_pred):
        """自定义评价函数
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


if __name__ == "__main__":
    print("start")
    a = time.time()
    lgb_params = {'objective': 'regression',
                  'boosting_type': 'gbdt',
                  'metric': 'mae',
                  'num_leaves': 63,
                  'max_bin': 70,
                  'learning_rate': 0.05,
                  'colsample_bytree': 0.8,
                  'bagging_fraction': 0.9,
                  'min_child_samples': 45,
                  'lambda_l1': 1,
                  'n_jobs': -1,
                  'seed': 1000}
    #     P = {"p1":lgb_params}
    train_model = Train_model()
    train_model.data()
    for target in [ "p1", "p2", "p3", "p4", "p5", "p6" ]:

        train_model.train(target, lgb_params)
    train_model.sub_result()
    print("time:{:6.4f} mins".format((time.time() - a) / 60))
