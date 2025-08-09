from google.colab import drive
import pandas as pd
import numpy as np

drive.mount('/content/drive')

path = "/content/drive/My Drive/CIC IDS main/"

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
nRowsRead = None

df1=pd.read_csv(path + "Monday-WorkingHours.pcap_ISCX.csv")
df2=pd.read_csv(path + "Tuesday-WorkingHours.pcap_ISCX.csv")
df3=pd.read_csv(path + "Wednesday-workingHours.pcap_ISCX.csv")
df4=pd.read_csv(path + "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
df5=pd.read_csv(path + "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
df6=pd.read_csv(path + "Friday-WorkingHours-Morning.pcap_ISCX.csv")
df7=pd.read_csv(path + "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df8=pd.read_csv(path + "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

df = pd.concat([df1,df2])
del df1,df2
df = pd.concat([df,df3])
del df3
df = pd.concat([df,df4])
del df4
df = pd.concat([df,df5])
del df5
df = pd.concat([df,df6])
del df6
df = pd.concat([df,df7])
del df7
df = pd.concat([df,df8])
del df8

nRow, nCol = df.shape
print(df.shape)

import pandas as pd
df['Flow Bytes/s'] = df['Flow Bytes/s'].fillna(0)

print(df.isnull().sum())

import pandas as pd

print("Shape before duplicate samples removal:", df.shape)

duplicate_samples = df[df.duplicated()]

print("Duplicate Samples Count:", len(duplicate_samples))

df = df.drop_duplicates()


label_mapping = {
    'FTP-Patator':'Brute Force',
    'SSH-Patator':'Brute Force',

    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'DoS Hulk': 'DoS',
    'Heartbleed':'DoS',
    'Bot':'DoS',

    'Web Attack � Brute Force': 'Web Attacks',
    'Web Attack � XSS': 'Web Attacks',
    'Web Attack � Sql Injection': 'Web Attacks',
}

df[' Label']=df[' Label'].replace(label_mapping)

sampling_strategy_over = {
    "Brute Force": 12000,                 
    "DoS": 205000,                  
    "Web Attacks": 10000,             
    "Infiltration": 1500,   
    "PortScan": 100000,              
    "DDoS": 150000,         
}

sampling_strategy_under = {
    "BENIGN": 500000                     
}

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

under_sampler = RandomUnderSampler(sampling_strategy=sampling_strategy_under)
over_sampler = RandomOverSampler(sampling_strategy=sampling_strategy_over)

X = df.drop(' Label', axis=1).values
y = df[' Label'].values

X_under, y_under = under_sampler.fit_resample(X, y)

X_resampled, y_resampled = over_sampler.fit_resample(X_under, y_under)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
from tqdm import tqdm

flow_text_train = [' '.join(map(str, row)) for row in X_train]

print("Number of Rows in flow_text_train:", len(flow_text_train))

X_train_tokenized = []
print("\nTokenizing the Training Set:")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') 
for text in tqdm(flow_text_train):
    tokenized_text = tokenizer.encode_plus(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_attention_mask=True,
        return_tensors='tf'
    )
    X_train_tokenized.append(tokenized_text)

X_train_input_ids = tf.squeeze(tf.stack([t['input_ids'] for t in X_train_tokenized]), axis=1)
X_train_attention_mask = tf.squeeze(tf.stack([t['attention_mask'] for t in X_train_tokenized]), axis=1)

print("Extracting Features from the Training Set:")
bert_model = TFBertModel.from_pretrained('bert-large-uncased')  
X_train_features = []
for i in tqdm(range(0, len(X_train_input_ids), 100)):
    batch_input_ids = X_train_input_ids[i:i+100]
    batch_attention_mask = X_train_attention_mask[i:i+100]
    features = bert_model([batch_input_ids, batch_attention_mask])[0][:, 0, :].numpy()
    X_train_features.append(features)
X_train_features = np.concatenate(X_train_features, axis=0)
print("Train features shape:", X_train_features.shape)

np.save(path +'train_features_BERT_Base.npy', X_train_features)


from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
from tqdm import tqdm

flow_text_test = [' '.join(map(str, row)) for row in X_test]

print("Number of Rows in flow_text_test:", len(flow_text_test))

X_test_tokenized = []
print("\nTokenizing the Testing Set:")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
for text in tqdm(flow_text_test):
    tokenized_text = tokenizer.encode_plus(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_attention_mask=True,
        return_tensors='tf'
    )
    X_test_tokenized.append(tokenized_text)

X_test_input_ids = tf.squeeze(tf.stack([t['input_ids'] for t in X_test_tokenized]), axis=1)
X_test_attention_mask = tf.squeeze(tf.stack([t['attention_mask'] for t in X_test_tokenized]), axis=1)

print("Extracting Features from the Testing Set:")
bert_model = TFBertModel.from_pretrained('bert-large-uncased')
X_test_features = []
for i in tqdm(range(0, len(X_test_input_ids), 100)):
    batch_input_ids = X_test_input_ids[i:i+100]
    batch_attention_mask = X_test_attention_mask[i:i+100]
    features = bert_model([batch_input_ids, batch_attention_mask])[0][:, 0, :].numpy()
    X_test_features.append(features)
X_test_features = np.concatenate(X_test_features, axis=0)
print("Test features shape:", X_test_features.shape)

np.save(path +'test_features_BERT_Base.npy', X_test_features)

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

def create_base_model(input_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,))
    ])
    return model

def add_classification_head(base_model, num_classes):
    classification_head = tf.keras.layers.Dense(num_classes, activation='softmax')
    model = tf.keras.models.Sequential([
        base_model,
        classification_head
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

train_features = np.load(path +'train_features_BERT_Base.npy')
train_labels = y_train

lookup_layer = tf.keras.layers.StringLookup(oov_token='<OOV>')
lookup_layer.adapt(train_labels)
train_labels_encoded = lookup_layer(train_labels)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = np.load(path +'test_features_BERT_Base.npy')
test_features = scaler.transform(test_features)
test_labels_encoded = lookup_layer(y_test)

base_model = create_base_model(train_features.shape[1])
num_classes_cicids = len(lookup_layer.get_vocabulary())
model_cicids = add_classification_head(base_model, num_classes_cicids)

model_cicids.fit(train_features, train_labels_encoded, epochs=5, batch_size=32, validation_split=0.1)

predictions = np.argmax(model_cicids.predict(test_features), axis=1)
decoded_predictions = [lookup_layer.get_vocabulary()[pred] for pred in predictions]

accuracy = accuracy_score(y_test, decoded_predictions)

print("CICIDS 2017 Accuracy:", accuracy)

base_model.save(path +'cicids_base_model.h5')
joblib.dump(scaler, path +'cicids_scaler.pkl')

lookup_config = lookup_layer.get_config()
lookup_weights = lookup_layer.get_weights()

joblib.dump(lookup_config, path +'lookup_config.pkl')
joblib.dump(lookup_weights, path +'lookup_weights.pkl')


model_architecture = model_cicids.to_json()
with open(path +'cicids_model_architecture.json', 'w') as json_file:
    json_file.write(model_architecture)
model_cicids.save_weights(path +'cicids_model_weights.h5')

report = classification_report(y_test, decoded_predictions)

print(report)

vocabulary_cicids = lookup_layer.get_vocabulary()
print(vocabulary_cicids)

np.save(path +'cicids_train_labels.npy', y_train)

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

with open(path +'cicids_model_architecture.json', 'r') as json_file:
    model_json = json_file.read()

model_cicids = tf.keras.models.model_from_json(model_json)

model_cicids.load_weights(path +'cicids_model_weights.h5')

scaler = joblib.load(path +'cicids_scaler.pkl')

test_features = np.load(path +'test_features_BERT_Base.npy')
test_features = scaler.transform(test_features)

lookup_config = joblib.load(path +'lookup_config.pkl')
lookup_weights = joblib.load(path +'lookup_weights.pkl')

lookup_layer = tf.keras.layers.StringLookup.from_config(lookup_config)
lookup_layer.set_weights(lookup_weights)

predictions = np.argmax(model_cicids.predict(test_features), axis=1)
decoded_predictions = [lookup_layer.get_vocabulary()[pred] for pred in predictions]

accuracy = accuracy_score(y_test, decoded_predictions)
print("CICIDS 2017 Accuracy:", accuracy)

report = classification_report(y_test, decoded_predictions)

print(report)

accuracy = accuracy_score(y_test, decoded_predictions)

report = classification_report(y_test, decoded_predictions, output_dict=True)

print("Classification Report:")
print(classification_report(y_test, decoded_predictions))

precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

print("Overall Accuracy:", accuracy)
print("Overall Precision:", precision)
print("Overall Recall:", recall)
print("Overall F1 Score:", f1_score)
