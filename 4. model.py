import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from collections import Counter
import matplotlib.pyplot as plt


def train(feats_arr, tag_arr, pos_neg_ratio=1/6):
    pred_result = []  # 保存五折交叉检验预测结果
    pred_result_novelty = []
    pred_result_impact = []
    pred_result_idx = []

    global result_prediction
    result_prediction = []

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    for train_idx, test_idx in kf.split(feats_arr):
        # print('train_index:%s , test_index: %s ' %(train_idx , test_idx))
        train_X, train_y = feats_arr[train_idx], tag_arr[train_idx]
        test_X, test_y = feats_arr[test_idx], tag_arr[test_idx]

        # 随机抽样
        # probability = sum(train_y) / (sum(1 - train_y) * pos_neg_ratio)
        # select_idx = []
        # for idx, t in enumerate(train_y.tolist()):
        #     if t == 1:
        #         select_idx.append(idx)
        #     elif t == 0:
        #         temp = random.uniform(0, 1)
        #         if temp <= probability:
        #             select_idx.append(idx)
        # train_X_rus, train_y_rus = train_X[select_idx], train_y[select_idx]
        # print(Counter(train_y_rus))

        # 使用原始数据，比例约为 1 : 7
        train_X_rus, train_y_rus = train_X, train_y

        """使用逻辑回归"""
        # 模型拟合  "balanced"
        # lr = LogisticRegression(class_weight=None, C=0.01)
        # # lr = SVC(probability=True, kernel="rbf", random_state=random_seed)
        # # lr = RandomForestClassifier(random_state=random_seed)
        #
        # # 使用全部指标进行估计
        # lr.fit(train_X_rus, train_y_rus)
        # pred_proba_lr = lr.predict_proba(test_X)[:, 1]  # 返回1的概率
        #
        # # 预测数据集
        # result_prediction.append(lr.predict_proba(X_for_predict)[:, 1])
        #
        # # 只使用创新指标进行估计
        # lr.fit(train_X_rus[:, [0, 1]], train_y_rus)
        # pred_proba_lr_novelty = lr.predict_proba(test_X[:, [0, 1]])[:, 1]
        #
        # # 只使用影响力指标进行估计
        # lr.fit(train_X_rus[:, 2:], train_y_rus)
        # pred_proba_lr_impact = lr.predict_proba(test_X[:, 2:])[:, 1]


        """使用xgboost"""
        xgb_model = xgb.XGBClassifier(booster='gbtree', objective='binary:logistic')
        xgb_model.fit(train_X_rus, train_y_rus)
        pred_proba_lr = xgb_model.predict_proba(test_X)[:, 1]

        # 预测数据集
        result_prediction.append(xgb_model.predict_proba(X_for_predict)[:, 1])   # 模型3

        xgb_model.fit(train_X_rus[:, [0, 1]], train_y_rus)
        pred_proba_lr_novelty = xgb_model.predict_proba(test_X[:, [0,1]])[:, 1]  # 模型1

        xgb_model.fit(train_X_rus[:, 2:], train_y_rus)
        pred_proba_lr_impact = xgb_model.predict_proba(test_X[:, 2:])[:, 1]      # 模型2


        pred_result.extend(pred_proba_lr.tolist())  # 样本预测值
        pred_result_novelty.extend(pred_proba_lr_novelty.tolist())
        pred_result_impact.extend(pred_proba_lr_impact.tolist())
        pred_result_idx.extend(test_idx)  # 样本编号，方便后续变回原来顺序
        assert len(pred_result) == len(pred_result_idx) == len(pred_result_novelty) == len(pred_result_impact)

    # 预测结果还原回原来的顺序
    global pred_arr
    pred_arr = np.zeros(tag_arr.shape)
    pred_arr_novelty = np.zeros(tag_arr.shape)
    pred_arr_impact = np.zeros(tag_arr.shape)
    for idx, predict, predict_novelty, predict_impact in zip(pred_result_idx, pred_result, pred_result_novelty,
                                                             pred_result_impact):
        pred_arr[idx] = predict
        pred_arr_novelty[idx] = predict_novelty
        pred_arr_impact[idx] = predict_impact

    # pd.DataFrame(data={"pred": pred_arr, "tag": tag_arr}).to_excel("train_output.xlsx")

    # 画 roc 曲线，并确定最佳阈值，同时返回该阈值下的分类报告
    draw_roc(tag_arr.copy(), pred_arr.copy())

    # 画决策曲线，并确定最佳阈值，同时返回该阈值下的分类报告
    draw_dc(tag_arr.copy(), pred_arr.copy(), pred_arr_novelty.copy(), pred_arr_impact.copy())


def draw_roc(true_tag_arr, predict_tag_arr):
    # 最大化 真阳-假阳的差值 ，其对应的阈值就是最佳阈值
    print(f"样本数 {true_tag_arr.shape[0]}，正样本数 {true_tag_arr.sum()}，负样本数 {(1 - true_tag_arr).sum()}")
    fpr, tpr, threshold = roc_curve(true_tag_arr, predict_tag_arr)
    roc_auc = auc(fpr, tpr)
    J = tpr - fpr
    idx = np.argmax(J)
    best_threshold = threshold[idx]
    print(f"ROC曲线对应的AUC值 {roc_auc: .4f}, ROC曲线确定的最优阈值 {best_threshold: .4f}\n")

    predict_tag_arr[predict_tag_arr >= best_threshold] = 1
    predict_tag_arr[predict_tag_arr < best_threshold] = 0
    print(f"按照最优阈值分类后的混淆矩阵：\n{confusion_matrix(true_tag_arr, predict_tag_arr)}\n")
    print(f"模型按照最优阈值分类的分类报告：\n{classification_report(true_tag_arr, predict_tag_arr)}")

    plt.figure()
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def draw_dc(true_tag_arr, predict_tag_arr_main, predict_tag_arr_novelty, predict_tag_arr_impact):
    pt_arr = []
    net_bnf_arr_main = []
    net_bnf_arr_novelty = []
    net_bnf_arr_impact = []
    jiduan = []
    for i in range(0, 100, 1):
        pt = i / 100
        pt_arr.append(pt)
        jiduan.append(
            (true_tag_arr.sum() - (true_tag_arr.shape[0] - true_tag_arr.sum()) * pt / (1 - pt)) / true_tag_arr.shape[0])

        # net_bnf_arr_main
        y_pred_lr = predict_tag_arr_main.copy()
        pred_ans_clip = np.zeros(y_pred_lr.shape[0])
        for j in range(y_pred_lr.shape[0]):
            if y_pred_lr[j] >= pt:
                pred_ans_clip[j] = 1
            else:
                pred_ans_clip[j] = 0
        TP = np.sum((true_tag_arr) * np.round(pred_ans_clip))
        FP = np.sum((1 - true_tag_arr) * np.round(pred_ans_clip))
        net_bnf = (TP - (FP * pt / (1 - pt))) / true_tag_arr.shape[0]
        net_bnf_arr_main.append(net_bnf)
        # print('pt {}, TP {}, FP {}, net_bf {}'.format(pt,TP,FP,net_bnf))

        # net_bnf_arr_novelty
        y_pred_lr = predict_tag_arr_novelty.copy()
        pred_ans_clip = np.zeros(y_pred_lr.shape[0])
        for j in range(y_pred_lr.shape[0]):
            if y_pred_lr[j] >= pt:
                pred_ans_clip[j] = 1
            else:
                pred_ans_clip[j] = 0
        TP = np.sum((true_tag_arr) * np.round(pred_ans_clip))
        FP = np.sum((1 - true_tag_arr) * np.round(pred_ans_clip))
        net_bnf = (TP - (FP * pt / (1 - pt))) / true_tag_arr.shape[0]
        net_bnf_arr_novelty.append(net_bnf)

        # net_bnf_arr_impact
        y_pred_lr = predict_tag_arr_impact.copy()
        pred_ans_clip = np.zeros(y_pred_lr.shape[0])
        for j in range(y_pred_lr.shape[0]):
            if y_pred_lr[j] >= pt:
                pred_ans_clip[j] = 1
            else:
                pred_ans_clip[j] = 0
        TP = np.sum((true_tag_arr) * np.round(pred_ans_clip))
        FP = np.sum((1 - true_tag_arr) * np.round(pred_ans_clip))
        net_bnf = (TP - (FP * pt / (1 - pt))) / true_tag_arr.shape[0]
        net_bnf_arr_impact.append(net_bnf)

    pt_return = []
    for a, b, c, d in zip(pt_arr, net_bnf_arr_main, net_bnf_arr_novelty, net_bnf_arr_impact):
        if b > c and b > d and b > 0:
            pt_return.append(a)
    print(f"根据决策曲线分析，可以取的阈值为：{pt_return}")

    plt.plot(pt_arr, net_bnf_arr_novelty, color='black', lw=1, linestyle='--', label='Model 1')
    plt.plot(pt_arr, net_bnf_arr_impact, color='black', lw=1, linestyle='-.', label='Model 2')
    plt.plot(pt_arr, net_bnf_arr_main, color='black', lw=1, linestyle='-', label='Model 3')

    plt.plot(pt_arr, np.zeros(len(pt_arr)), color='black', lw=1, linestyle='dotted', label='Treat None')
    plt.plot(pt_arr, jiduan, color='black', lw=1, linestyle='dotted', label='Treat All')
    plt.xlim([0, 1])
    plt.ylim([-0.25, 0.25])
    plt.xlabel('分类阈值')
    plt.ylabel('净收益')
    plt.title('决策曲线分析（DCA）')
    plt.legend(loc="lower right")
    plt.savefig("DCA.png")
    plt.show()


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    rs = 59  # random_seed 保证结果可复现
    random_seed = rs
    random.seed(rs)

    # 读取数据
    data = pd.read_excel("./model_data.xlsx", sheet_name=1)
    data.columns = ["patent_id", 'F1', 'F2', "F3", "F4", "F5", "tag"]
    print(data.describe().round(4))

    # 数据标准化
    # feats = np.array(data.iloc[:, [1, 2, 3, 4, 9]])
    feats = np.array(data.iloc[:, [1, 2, 3, 4, 5]])
    tag = np.array(data.iloc[:, -1])
    feats_std = (feats - feats.mean(axis=0)) / feats.std(axis=0)

    # 模型数据集
    X = feats_std[:824, :]
    y = tag[:824]
    print(f"样本量 {y.shape[0]}，特征数 {X.shape[1]}，正负样本比例 {int(sum(y))}：{int(sum(1 - y))}")

    X_for_predict = feats_std[824:, :]
    print(f"预测集样本量 {X_for_predict.shape[0]}，特征数 {X_for_predict.shape[1]}")

    pat_id_for_prediction = data.iloc[824:, 0]

    train(X, y)

    input_text = str(input("输入合适阈值，结束请输入q"))
    while input_text.lower() != "q":
        temp = pred_arr.copy()
        threshold = float(input_text)
        temp[temp >= threshold] = 1
        temp[temp < threshold] = 0
        print(confusion_matrix(y, temp))
        print(classification_report(y, temp))
        input_text = str(input("输入合适阈值，结束请输入q"))

    proba = 0
    for temp in result_prediction:
        proba += temp
    proba /= 5
    result_df = pd.DataFrame(pat_id_for_prediction[proba >= float(input("请输入最终预测的阈值"))])
    result_df.to_excel('patent_disruption_result.xlsx')
    print(f"1997-2014年模型预测的颠覆性专利数量：{len(result_df)}")






