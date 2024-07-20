import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import RDLogger
from tqdm import tqdm
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import GridSearchCV

RDLogger.DisableLog('rdApp.*')
###################################################
# 全局变量
file_path = './dataset/pubchem_Q9NZQ7_smiles_pIC50.csv'
morgan_radius = 3
morgan_hashsize = 1024
normalize = True
method = 'MLP'
units = (300, 300, 300)
###################################################
df = pd.read_csv(file_path)
# smiles_format = df.iloc[:, 0].tolist()
# activity_value = np.array(df.iloc[:, 1].tolist())
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

X_train, X_test, y_train, y_test = train_test_split(df_fingerprint, activity_value, test_size=0.30, random_state=42)

# 调参
mlp = MLPRegressor(random_state=42)
mlp_tuned_parameters = {"hidden_layer_sizes": [(100,), (200,), (300,), (400,), (500,)]}
# mlp_tuned_parameters = {"learning_rate_init": [0.0001, 0.001, 0.01, 0.1]}
# mlp_tuned_parameters = {"hidden_layer_sizes": [(100,), (100, 100), (100, 100, 100), (100, 100, 100, 100)]}
estimator = GridSearchCV(mlp, mlp_tuned_parameters, cv=5)
estimator.fit(X_train, y_train)
print(
    "The best parameters are %s with a score of %0.2f"
    % (estimator.best_params_, estimator.best_score_)
    )

# 模型拟合
nn = MLPRegressor(**estimator.best_params_, random_state=42)
# nn = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100), random_state=42)
nn.fit(X_train, y_train)
joblib.dump(nn, 'MLP/MLP.pkl')

y_train_pred = nn.predict(X_train)
y_test_pred = nn.predict(X_test)

r2_train  = r2_score(y_train, y_train_pred)
r2_test  = r2_score(y_test, y_test_pred)


print(method, "test set MAE", mean_absolute_error(y_test, y_test_pred))
print(method, "test set MSE:", mean_squared_error(y_test, y_test_pred))
print(method, "test set RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print(method, 'training set R2 score: {:.3f}'.format(r2_train))
print(method, 'test set R2 score: {:.3f}'.format(r2_test))

# 模型预测图像 train set
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, color='green')
plt.ylabel('Predictive pIC50', fontdict={'fontsize': 15})
plt.xlabel('Experimental pIC50', fontdict={'fontsize': 15})
plt.axis([-4.0, 4.0, -4.0, 4.0])
plt.axline((0.0, 0.0), slope=1.0, color='gray', lw=3, linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig('MLP/MLP_train_set.png')
plt.close()

# 模型预测图像 test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='green')
plt.ylabel('Predictive pIC50', fontdict={'fontsize': 15})
plt.xlabel('Experimental pIC50', fontdict={'fontsize': 15})
plt.axis([-4.0, 4.0, -4.0, 4.0])
plt.axline((0.0, 0.0), slope=1.0, color='gray', lw=3, linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig('MLP/MLP_test_set.png')
plt.close()

# X_sample = shap.sample(X_test, 100)
# explainer = shap.KernelExplainer(nn.predict, X_sample)
# shap_values = explainer.shap_values(X_sample)
# 
# img1 = shap.summary_plot(shap_values=shap_values, features=X_sample, feature_names=df_fingerprint.columns, show=False)
# plt.savefig('MLP_model/img1.png')
# plt.close()
# img2 = shap.summary_plot(shap_values=shap_values, features=X_sample, feature_names=df_fingerprint.columns, plot_type='bar', show=False)
# plt.savefig('MLP_model/img2.png')
# plt.close()