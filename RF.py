import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import shap
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import RDLogger
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_histogram
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
###################################################
# 全局变量
file_path = './dataset/pubchem_Q9NZQ7_smiles_pIC50.csv'
morgan_radius = 3
morgan_hashsize = 1024
normalize = True
method = 'RF'
###################################################
df = pd.read_csv(file_path)
smiles_format = df.iloc[:, 2].tolist()
activity_value = np.array(df.iloc[:, 3].tolist())

def morgan_from_list(smiles, radius, nbits):
    """
    接收smiles分子结构列表，得到每个分子对应的分子指纹向量和结构信息
    :param molecules: 分子结构的列表
    :param radius: Morgan指纹的半径
    :param nbits: Morgan指纹的哈希长度
    :return: 分子指纹向量（fingerprint）和指纹结构信息（bit_information）
    """
    molecules = [Chem.MolFromSmiles(s) for s in smiles]
    bit_information = []
    fingerprint = []
    for molecule in molecules:
        bi = {}
        finger = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nbits, bitInfo=bi)
        finger_array = np.zeros((1,), dtype=np.int8)
        ConvertToNumpyArray(finger, finger_array)
        fingerprint.append(finger_array)
        bit_information.append(bi)
    
    dataframe_fingerprint = pd.DataFrame()
    
    print(f"Start generating the morgan fingerprints with parameters(radius:{radius}, hashsize:{nbits})")
    for fp in tqdm(fingerprint):
        df_fp = pd.DataFrame(fp)
        df_fp = df_fp.T
        dataframe_fingerprint = pd.concat([dataframe_fingerprint, df_fp], axis=0)

    return dataframe_fingerprint, bit_information

df_fingerprint, bit_info = morgan_from_list(smiles_format, morgan_radius, morgan_hashsize)

# Z-score Gaussian normalization
if normalize:
    mean = np.mean(activity_value)
    std = np.std(activity_value)
    activity_value = (activity_value - mean) / std
    print()
    print(f"Z-score normalization has been done, mean is {mean}, std is {std}.")
    print(activity_value)

# 拆分训练，验证和测试集
X_train, X_test, y_train, y_test = train_test_split(df_fingerprint, activity_value, test_size=0.30, random_state=5)
# X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.50, random_state=5)

# (parameter space, # of evaluations)
opt = BayesSearchCV(RandomForestRegressor(criterion='squared_error',random_state=5),
    {
        'n_estimators': Integer(100, 500),
        'max_depth': Integer(1, 20),
        'max_features': Integer(5, 30),
        'min_samples_leaf': Integer(1, 10),
        'min_samples_split': Integer(2, 21),
    },
    n_iter=40,
    cv=5
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
print("best params: %s" % str(opt.best_params_))

# 模型拟合
reg = RandomForestRegressor(**opt.best_params_, criterion='squared_error', random_state=5)

reg.fit(X_train, y_train)

# 保存模型
joblib.dump(reg, 'RF/RF.pkl')

# 模型评估
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

r2_train  = r2_score(y_train, y_train_pred)
r2_test  = r2_score(y_test, y_test_pred)

print(method, "test set MAE", mean_absolute_error(y_test, y_test_pred))
print(method, "test set MSE:", mean_squared_error(y_test, y_test_pred))
print(method, "test set RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print(method, 'training set R2 score: {:.3f}'.format(r2_train))
print(method, 'test set R2 score: {:.3f}'.format(r2_test))

# 模型预测图像 test set
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, color='red')
plt.ylabel('Predictive pIC50', fontdict={'fontsize': 15})
plt.xlabel('Experimental pIC50', fontdict={'fontsize': 15})
plt.axis([-4.0, 4.0, -4.0, 4.0])
plt.axline((0.0, 0.0), slope=1.0, color='gray', lw=3, linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig('RF/RF_train_set.png')
plt.close()

# 模型预测图像 test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='red')
plt.ylabel('Predictive pIC50', fontdict={'fontsize': 15})
plt.xlabel('Experimental pIC50', fontdict={'fontsize': 15})
plt.axis([-4.0, 4.0, -4.0, 4.0])
plt.axline((0.0, 0.0), slope=1.0, color='gray', lw=3, linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig('RF/RF_test_set.png')
plt.close()

# X_sample = shap.sample(X_test, 100)
# explainer = shap.TreeExplainer(reg, data=X_sample)
# shap_values = explainer.shap_values(X_sample)
# print('base value:', explainer.expected_value)
# img1 = shap.summary_plot(shap_values=shap_values, features=X_sample, feature_names=df_fingerprint.columns, show=False)
# plt.savefig('RF_model/img1-100_newdataset.png')
# plt.close()
# 
# img2 = shap.summary_plot(shap_values=shap_values, features=X_sample, feature_names=df_fingerprint.columns, plot_type='bar', show=False)
# plt.savefig('RF_model/img2-100_newdataset.png')
# plt.close()

# img3 = shap.plots.force(explainer.expected_value, shap_values[0], matplotlib=True, show=False)
# plt.savefig('img.png', bbox_inches='tight', dpi=300)
# plt.close()
