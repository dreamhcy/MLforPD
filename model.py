import csv
import os
from collections import Counter
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from utils.util import *
from optimization import *
import warnings
from sklearn.linear_model import Lasso
from ReliefF import ReliefF
warnings.filterwarnings("ignore")



class ML():
    def __init__(self, fold_path, data_name, taskid, modality, all, k, top_n, seed, classifier, select, optimization=None, logger=None):
        self.fold_path = fold_path
        self.modality = modality
        self.all = all
        self.k_folds = k
        self.taskid = taskid
        self.ids, self.datas, self.feature_names, self.y = self.get_data(data_name)
        self.top_n = top_n
        self.seed = seed
        self.optimization = optimization
        self.classifier = classifier
        self.select = select
        self.logger = logger

    def get_data(self, dataset):

        data_path = f'../coa_ml_parkinson/features1/feature_{dataset}'

        feature_names, datas = {f: [] for f in self.modality}, {f: [] for f in self.modality}
        for f in self.modality:
            if f == 'scalar':
                feature_path = data_path + f'/{f}/all/feature{self.taskid}.csv'
            else:
                if self.all:
                    feature_path = data_path + f'/{f}/all/feature{self.  taskid}.csv'
                else:
                    feature_path = data_path + f'/{f}/common/feature{self.taskid}.csv'
            datas[f] = pd.read_csv(feature_path, header=0)
            feature_names[f] = np.array(list(datas[f].drop(columns=['id', 'label']).columns))
            labels = np.array(datas[f])[:, -1]
            ids = np.array(datas[f])[:, 0]
            datas[f] = np.array(datas[f])[:, 1:-1]
        return ids, datas, feature_names, labels

    def feature_selection(self, X_train, y_train):
        if self.select.upper() == 'RandomForest'.upper():
            rf = RandomForestClassifier(n_estimators=100, random_state=self.seed)
            # 拟合模型
            rf.fit(X_train, y_train)
            # 获取特征重要性
            feature_importance = rf.feature_importances_
        elif self.select.upper() == 'relief'.upper():
            # 使用ReliefF进行特征选择
            relief = ReliefF(n_neighbors=10, n_features_to_keep=self.top_n)  # n_neighbors是算法参数，可以调节
            relief.fit(X_train, y_train)

            # 获取特征的重要性
            feature_importance = relief.feature_scores
        else:
            # 使用Lasso进行特征选择
            lasso = Lasso(alpha=0.01, random_state=self.seed)  # alpha是L1正则化的参数，可以调节
            lasso.fit(X_train, y_train)

            # 获取特征的重要性（绝对值）
            feature_importance = np.abs(lasso.coef_)

        num = min(X_train.shape[1], self.top_n)
        # 对特征重要性进行排序
        selected_features = np.argsort(feature_importance)[-num:]

        return selected_features


    def fit(self):
        # 最大最小归一化
        self.datas = {f: MinMaxScaler().fit_transform(self.datas[f]) for f in self.datas.keys()}
        # 开始训练
        # 初始化 StratifiedKFold 对象
        self.logger.info(f'start {self.k_folds}-folds cross validation ……')
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)

        # 遍历每个折
        count = 1
        result_fold = []
        for train_index, valid_index in kfold.split(self.y):

            datas_train, datas_valid = ({f: self.datas[f][train_index] for f in self.datas.keys()},
                                                 {f: self.datas[f][valid_index] for f in self.datas.keys()})
            y_train, y_valid = self.y[train_index], self.y[valid_index]
            valid_ids = self.ids[valid_index]
            # 预测
            y_pred = self.one_fold_process(datas_train, datas_valid, y_train, y_valid, valid_ids, count)

            # 计算评估指标
            cm = confusion_matrix(y_valid, y_pred)
            report = classification_report(y_valid, y_pred)
            accuracy = round(accuracy_score(y_valid, y_pred), 4)
            precision = np.around(precision_score(y_valid, y_pred, average=None), 4)
            recall = np.around(recall_score(y_valid, y_pred, average=None), 4)
            spe = np.around(cal_Specificity(y_valid, y_pred, average=None), 4)
            f1 = np.around(f1_score(y_valid, y_pred, average=None), 4)
            auc = np.around(
                roc_auc_score(self.to_one_hot(y_valid), self.to_one_hot(y_pred), average=None, multi_class='ovo'),
                4)
            result_k = [self.seed, self.taskid, self.classifier, count, accuracy] + list(precision) + list(
                recall) + list(spe) + list(f1) + list(auc)
            self.logger.info(f'The confusion matrix of the {count}th fold is{cm}')
            self.logger.info(f'The classification report of the {count}th fold is{report}')
            self.logger.info(f'The result of the {count}th fold is{result_k}')
            # 记录每折的评估结果
            result_fold.append(result_k)
            # 打开CSV文件，使用'w'模式创建一个新文件，如果文件已存在则覆盖
            with open(os.path.join(self.fold_path, "evaluation_results_per_fold.csv"), mode='a', newline='') as file:
                writer = csv.writer(file)
                # 写入数据到CSV文件
                writer.writerow(result_k)

            count += 1
        # 获取最终指标--均值±（标准差）
        result_fold = np.array(result_fold)

        mean = [self.seed, self.taskid, self.classifier] + list(np.mean(np.array(result_fold[:, 4:], dtype=float), axis=0))
        std = [self.seed, self.taskid, self.classifier] + list(np.std(np.array(result_fold[:, 4:], dtype=float), axis=0))


        return mean, std

    def to_one_hot(self, y):
        y = np.array(y)
        y_onehot = np.zeros((y.size, 3))
        y_onehot[np.arange(y.size).astype(int), y.astype(int)] = 1
        return y_onehot



    def one_fold_process(self, datas_train, datas_valid, y_train, y_valid, valid_ids, count):
        # 特征选择
        y_preds, y_probs = {f: [] for f in self.modality}, {f: [] for f in self.modality}
        weight, score_sum = {f: 0 for f in self.modality}, 0.0
        c_count = np.zeros((len(y_valid), 3))
        for f in self.modality:
            self.logger.info(f"start train {f} modality……")
            # 特征选择
            selected_features = self.feature_selection(datas_train[f], y_train)
            # 保存特征选择结果
            item = {'fold': count, 'modality': f, 'selected_features': selected_features}
            with open(self.fold_path + f'/feature_select.txt', 'a') as ff:
                ff.write(f"{item['fold']} '{item['modality']}' {item['selected_features']}\n")
            selected_feature_names = self.feature_names[f][selected_features]
            self.logger.info(f'the selected features are {selected_feature_names}')
            datas_train[f] = datas_train[f][:, selected_features]
            datas_valid[f] = datas_valid[f][:, selected_features]
            y_pred_f, y_prob_f, clf, score = self.one_feature_train(datas_train[f], y_train, datas_valid[f], y_valid)
            # 保存模型
            joblib.dump(clf, self.fold_path + f'/{self.taskid}_{count}_{f}_model.joblib')
            y_preds[f], y_probs[f], weight[f] = y_pred_f, y_prob_f, score
            score_sum += score
            #
            y_pred_f = np.array(y_pred_f)
            y_onehot = np.zeros((y_pred_f.size, 3))
            y_onehot[np.arange(y_pred_f.size).astype(int), y_pred_f.astype(int)] = 1
            c_count += y_onehot

        # 决策级融合
        # 投票
        y_pred, y_prob = [], []
        for i in range(len(y_valid)):
            p = np.array([c_count[i, y_preds[f][i].astype(int)]/np.sum(c_count[i]) * y_probs[f][i] for f in y_probs.keys()])
            p = np.sum(p, axis=0)
            y_prob.append(p)
            y_pred.append(np.argmax(p))

        # 存储每折的预测概率
        # 将每个键对应的值转换为DataFrame并存储到列表中
        if len(np.unique(self.y)) == 2:
            dfs = [pd.DataFrame(value, columns=[f'{key}_prob_hc', f'{key}_prob_pd']) for key, value in
                   y_probs.items()]
        else:
            dfs = [pd.DataFrame(value, columns=[f'{key}_prob_hc', f'{key}_prob_pd', f'{key}_prob_et']) for
                   key, value in y_probs.items()]
        # 存储ids和label
        dfs.insert(0, pd.DataFrame(np.column_stack((valid_ids, y_valid)), columns=['id', 'label']))
        #　存储最终概率值
        dfs.insert(len(dfs), pd.DataFrame(np.array(y_prob), columns=['prob_hc', 'prob_pd', 'prob_et']))
        # 使用pd.concat按列堆叠DataFrame
        prob = pd.concat(dfs, axis=1)
        prob.to_csv(os.path.join(self.fold_path, f'{self.taskid}_{self.classifier}_{count}_prob.csv'))

        return y_pred


    def one_feature_train(self, X_train, y_train, X_test, y_test):

        if self.optimization is not None:
            self.logger.info(f'Select the best params...............')
            org = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test,
                   'y_test': y_test, 'classifier': self.classifier, 'seed': self.seed}
            lb, ub, dim = function_params(self.classifier)
            best_param = self.optimization(classifer_objective_function, lb, ub, dim,
                                           50, 10, org)
            best_params = best_param.bestIndividual
            self.logger.info(f'the best params are {best_params}')
            # 训练模型
            if self.classifier == 'svm':
                clf = SVC(C=best_params[0], gamma=best_params[1], decision_function_shape='ovo', random_state=self.seed, probability=True)
            elif self.classifier == 'logistic':
                clf = LogisticRegression(C=best_params[0], multi_class='ovo', random_state=self.seed)
            elif self.classifier == 'xgboost':
                # 创建XGBoost分类器
                clf = xgb.XGBClassifier(n_estimators=int(best_params[0]), learning_rate=best_params[1],
                                        max_depth=int(best_params[2]), random_state=self.seed)
            else:
                base = DecisionTreeClassifier(max_depth=int(best_params[0]), min_samples_split=best_params[1],
                                              random_state=self.seed)
                clf = AdaBoostClassifier(base_estimator=base, n_estimators=int(best_params[2]), learning_rate=best_params[3],
                                         random_state=self.seed)
        else:
            if self.classifier == 'svm':
                clf = SVC(decision_function_shape='ovr', random_state=self.seed, probability=True)
            elif self.classifier == 'logistic':
                clf = LogisticRegression(multi_class='ovr', random_state=self.seed)
            elif self.classifier == 'xgboost':
                # 创建XGBoost分类器
                clf = xgb.XGBClassifier(random_state=self.seed)
            else:
                clf = AdaBoostClassifier(random_state=self.seed)

        clf.fit(X_train, y_train)

        # 预测
        y_test_pred = clf.predict(X_test)
        y_test_prob = clf.predict_proba(X_test)  # 2*n
        score = accuracy_score(y_test, y_test_pred)

        return y_test_pred, y_test_prob, clf, score

def init_save(fold_path, data_name):
    if data_name == 'PaHaW':
        evaluate_name = ['seed', 'taskid', 'classifier', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']
    else:
        evaluate_name = ['seed', 'taskid', 'classifier',
                         'accuracy', 'precision_class_hc', 'precision_class_pd',
                         'precision_class_et', 'recall_class_hc', 'recall_class_pd', 'recall_class_et',
                         'specificity_hc', 'specificity_pd', 'specificity_et',
                         'f1_class_hc', 'f1_class_pd', 'f1_class_et', 'auc_class_hc', 'auc_class_pd',
                         'auc_class_et']
    with open(os.path.join(fold_path, "evaluation_results_per_fold.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入数据到CSV文件
        if data_name == 'PaHaW':
            writer.writerow(['seed', 'taskid', 'classifier', 'fold', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc'])
        else:
            writer.writerow(['seed', 'taskid', 'classifier', 'fold', 'accuracy', 'precision_class_hc',
                             'precision_class_pd', 'precision_class_et', 'recall_class_hc', 'recall_class_pd', 'recall_class_et',
                             'specificity_hc', 'specificity_pd', 'specificity_et',
                             'f1_class_hc', 'f1_class_pd', 'f1_class_et', 'auc_class_hc', 'auc_class_pd', 'auc_class_et'])

    with open(os.path.join(fold_path, "evaluation_results_mean.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入数据到CSV文件
        writer.writerow(evaluate_name)
    with open(os.path.join(fold_path, "evaluation_results_std.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入数据到CSV文件
        writer.writerow(evaluate_name)


if __name__ == '__main__':

    f = open('config.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)


    optimization = rCOA
    opt = str(optimization).split(' ')[1]
    data_name = 'Two'

    fold_path = (f"../coa_ml_parkinson/results/"
                 f"{data_name}+{config['seed']}+{config['modality']}+{config['all']}+{config['select']}+"
                 f"{config['classifier']}+{config['top_n']}+{opt}+{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(fold_path, exist_ok=True)
    logger = get_logger(os.path.join(fold_path, f'logs.log'))
    init_save(fold_path, data_name)

    # 保存配置文件
    with open(os.path.join(fold_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # optimization = rCOA
    for taskid in tqdm([7]):
        logger.info(f'Select {taskid} task')
        logger.info(f'Select {optimization} optimization')

        classifier_results = ML(fold_path=fold_path, data_name=data_name, taskid=taskid,
                                modality=config['modality'], all=config['all'], k=config['k'],seed=config['seed'],
                                top_n=config['top_n'], classifier=config['classifier'], select=config['select'],
                                optimization=optimization, logger=logger)

        mean, std = classifier_results.fit()
        with open(os.path.join(fold_path, "evaluation_results_mean.csv"), mode='a', newline='') as file:
            writer = csv.writer(file)
            # 写入数据到CSV文件
            writer.writerow(mean)
        with open(os.path.join(fold_path, "evaluation_results_std.csv"), mode='a', newline='') as file:
            writer = csv.writer(file)
            # 写入数据到CSV文件
            writer.writerow(std)



