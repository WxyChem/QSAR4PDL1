import numpy as np
import pandas as pd
import keras
from keras import layers
from keras import optimizers
from keras.models import Sequential
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit import RDLogger
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

RDLogger.DisableLog('rdApp.*')

file_path = './dataset/pubchem_Q9NZQ7_smiles_pIC50.csv'
morgan_radius = 3
morgan_hashsize = 1024
normalize = True
method = 'LSTM'
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
# df_fingerprint = df_fingerprint.sample(n=724, axis=1)
# print(df_fingerprint.shape)

# Z-score Gaussian normalization
if normalize:
    mean = np.mean(activity_value)
    std = np.std(activity_value)
    activity_value = (activity_value - mean) / std
    print()
    print(f"Z-score normalization has been done, mean is {mean}, std is {std}.")
    print(activity_value)

X_train, X_test, y_train, y_test = train_test_split(df_fingerprint, activity_value, test_size=0.30, random_state=42)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model = Sequential()
model.add(layers.LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(layers.Dropout(0.1))
model.add(layers.LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(1, activation='linear'))
model.compile(loss='mse', optimizer=adam)

history = model.fit(X_train, y_train, batch_size=32, epochs=40, validation_data=(X_test, y_test))

model.save('LSTM/LSTM.h5')
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
r2_train  = r2_score(y_train, y_train_pred)
r2_test  = r2_score(y_test, y_test_pred)

print(method, 'Train set R2 score: {:.3f}'.format(r2_train))
print(method, 'test set R2 score: {:.3f}'.format(r2_test))
print(method, "test set MAE", mean_absolute_error(y_test, y_test_pred))
print(method, "test set MSE:", mean_squared_error(y_test, y_test_pred))
print(method, "test set RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))

plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, color='blue')
plt.ylabel('Predictive pIC50', fontdict={'fontsize': 15})
plt.xlabel('Experimental pIC50', fontdict={'fontsize': 15})
plt.axis([-4.0, 4.0, -4.0, 4.0])
plt.axline((0.0, 0.0), slope=1.0, color='gray', lw=3, linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig('LSTM/LSTM_model_train_set.png')
plt.close()
  
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.ylabel('Predictive pIC50', fontdict={'fontsize': 15})
plt.xlabel('Experimental pIC50', fontdict={'fontsize': 15})
plt.axis([-4.0, 4.0, -4.0, 4.0])
plt.axline((0.0, 0.0), slope=1.0, color='gray', lw=3, linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig('LSTM/LSTM_model_test_set.png')
plt.close()