from google.colab import drive
import pandas as pd
import numpy as np

drive.mount('/content/drive')
path = "/content/drive/My Drive/"

import time
import torch     
import torch.nn as nn             
import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt
import psutil
import subprocess
import ipywidgets as widgets
from IPython.display import display
from IPython.display import Markdown
import gc
gc.collect()
!pip install transformers
!pip install pandas
!pip install torch
!pip install tqdm
from transformers import BertModel, BertTokenizer    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split 
from tqdm import tqdm
from tqdm.notebook import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

train_url = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Train.csv'
test_url = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Test.csv'

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

df_train = pd.read_csv(train_url,header=None, names = col_names)
df_test = pd.read_csv(test_url, header=None, names = col_names)

df_train_label = df_train['label']
df_test_label = df_test['label']

df_train_label = df_train_label.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})

df_test_label = df_test_label.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})

df_train['label'] = df_train_label
df_test['label'] = df_test_label

Normal_Samples_Train = df_train[df_train['label'] == 0]['label'].count()
print("Normal labels/samples with value 0 in train dataset:", Normal_Samples_Train)
DoS_Samples_Train = df_train[df_train['label'] == 1]['label'].count()
print("DoS labels/samples with value 1 in train dataset:", DoS_Samples_Train)
Probe_Samples_Train = df_train[df_train['label'] == 2]['label'].count()
print("Probe labels/samples with value 2 in train dataset:", Probe_Samples_Train)
R2L_Samples_Train = df_train[df_train['label'] == 3]['label'].count()
print("R2L labels/samples with value 3 in train dataset:", R2L_Samples_Train)
U2R_Samples_Train = df_train[df_train['label'] == 4]['label'].count()
print("U2R labels/samples with value 4 in train dataset:", U2R_Samples_Train)
print("---------------------------------------------------------")
Normal_Samples_Test = df_test[df_test['label'] == 0]['label'].count()
print("Normal labels/samples with value 0 in test dataset:", Normal_Samples_Test)
DoS_Samples_Test = df_test[df_test['label'] == 1]['label'].count()
print("DoS labels/samples with value 1 in test dataset:", DoS_Samples_Test)
Probe_Samples_Test = df_test[df_test['label'] == 2]['label'].count()
print("Probe labels/samples with value 2 in test dataset:", Probe_Samples_Test)
R2L_Samples_Test = df_test[df_test['label'] == 3]['label'].count()
print("R2L labels/samples with value 3 in test dataset:", R2L_Samples_Test)
U2R_Samples_Test = df_test[df_test['label'] == 4]['label'].count()
print("U2R labels/samples with value 4 in test dataset:", U2R_Samples_Test)

import pandas as pd
from sklearn.model_selection import train_test_split

X_train = df_train.drop('label', axis=1)
y_train = df_train['label']
X_test = df_test.drop('label', axis=1)
y_test = df_test['label']

train_label_counts = pd.Series(y_train).value_counts()
test_label_counts = pd.Series(y_test).value_counts()


import pandas as pd
train_data = pd.concat([X_train, y_train], axis=1)

test_data = pd.concat([X_test, y_test], axis=1)

combined_data = pd.concat([train_data, test_data], axis=0)

X_combined = combined_data.drop(columns=['label'])  
y_combined = combined_data['label'] 

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

sampling_strategy_over = {
    3: 8000,  
    4: 500,   
}

over_sampler = RandomOverSampler(sampling_strategy=sampling_strategy_over)

oversampling_pipeline = Pipeline([
    ('over_sampling', over_sampler)
])


X_combined_resampled, y_combined_resampled = oversampling_pipeline.fit_resample(X_combined, y_combined)


X_train, X_test, y_train, y_test = train_test_split(X_combined_resampled, y_combined_resampled, test_size=0.2, random_state=42, stratify=y_combined_resampled)


train_label_counts = pd.Series(y_train).value_counts()
test_label_counts = pd.Series(y_test).value_counts()

#Train Set

from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
from tqdm import tqdm

flow_text_train = X_train.apply(lambda x: ' '.join(map(str, x)), axis=1)

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

np.save(path+'train_features_BERT_Large.npy', X_train_features)

#Test Set
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
from tqdm import tqdm


flow_text_test = X_test.apply(lambda x: ' '.join(map(str, x)), axis=1)

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

np.save(path+'test_features_BERT_Large.npy', X_test_features)

import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

train_features = np.load(path+'train_features_BERT_Large.npy')
train_labels = y_train

test_features = np.load(path+'test_features_BERT_Large.npy')
test_labels = y_test

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

scaler_filename = path+'training_scaler.joblib'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved as {scaler_filename}")

mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
mlp_classifier.fit(train_features_scaled, train_labels)
class_probabilities = mlp_classifier.predict_proba(test_features_scaled)
predictions = np.argmax(class_probabilities, axis=1)
accuracy = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions, average='weighted')
precision = precision_score(test_labels, predictions, average='weighted')
recall = recall_score(test_labels, predictions, average='weighted')
roc_auc = roc_auc_score(test_labels, class_probabilities, multi_class='ovr')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

cm = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

class_names = np.unique(test_labels).astype(str)
report = classification_report(test_labels, predictions, target_names=class_names)
print("Classification Report:")
print(report)

accuracy = accuracy_score(test_labels, predictions)

cm = confusion_matrix(test_labels, predictions)
class_count = len(np.unique(test_labels))

precision = np.zeros(class_count)
recall = np.zeros(class_count)
f1 = np.zeros(class_count)

for i in range(class_count):
    true_positive = cm[i, i]
    false_positive = sum(cm[:, i]) - true_positive
    false_negative = sum(cm[i, :]) - true_positive

    precision[i] = true_positive / (true_positive + false_positive)
    recall[i] = true_positive / (true_positive + false_negative)
    f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

roc_auc = roc_auc_score(test_labels, class_probabilities, multi_class='ovr')

for i in range(class_count):
    class_name = class_names[i]
    print(f"Class {class_name} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1 Score: {f1[i]:.4f}")

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

classifier_filename = path+'mlp_classifier.joblib'
joblib.dump(mlp_classifier, classifier_filename)
print(f"MLP classifier saved as {classifier_filename}")

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

test_features = np.load(path+'test_features_BERT_Large.npy')
test_labels = y_test

classifier_filename = path+'mlp_classifier.joblib'
mlp_classifier = joblib.load(classifier_filename)
scaler_filename = path+'training_scaler.joblib' 
scaler = joblib.load(scaler_filename)
test_features_scaled = scaler.transform(test_features)
class_probabilities = mlp_classifier.predict_proba(test_features_scaled)
predictions = np.argmax(class_probabilities, axis=1)
accuracy = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions, average='weighted')
precision = precision_score(test_labels, predictions, average='weighted')
recall = recall_score(test_labels, predictions, average='weighted')
roc_auc = roc_auc_score(test_labels, class_probabilities, multi_class='ovr')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")


class_labels = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}
cm = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(5, 4))
sns.heatmap(cm / cm.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2%', cmap='Blues', xticklabels=class_labels.values(), yticklabels=class_labels.values())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

class_names = np.unique(test_labels).astype(str)
report = classification_report(test_labels, predictions, target_names=class_names)
print("Classification Report:")
print(report)

for i in range(len(class_names)):
    accuracy_class = accuracy_score(test_labels[test_labels == i], predictions[test_labels == i])
    print(f"Accuracy for {class_names[i]}: {accuracy_class:.4f}")

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision = dict()
recall = dict()
thresholds = dict()

for i in range(len(class_labels)):
    precision[i], recall[i], thresholds[i] = precision_recall_curve(
        test_labels == i, class_probabilities[:, i])

plt.figure(figsize=(5, 4))
plt.xlabel('Recall')
plt.ylabel('Precision')

zoom_x_start = 0.5  
zoom_x_end = 1.0   
zoom_y_start = 0.9  
zoom_y_end = 1.0   

for i in range(len(class_labels)):
    plt.plot(recall[i], precision[i], label=f'{class_labels[i]}')

plt.xlim(zoom_x_start, zoom_x_end)
plt.ylim(zoom_y_start, zoom_y_end)
plt.legend(loc='best')
plt.tight_layout() 
plt.savefig(path+'precision_recall_curves_zoomed.pdf', format='pdf')
plt.show()
plt.close()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(class_labels)):
    fpr[i], tpr[i], _ = roc_curve(test_labels == i, class_probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(5, 4))  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


zoom_x_start = 0.0  
zoom_x_end = 1.0    
zoom_y_start = 0.997 
zoom_y_end = 1.0  

for i in range(len(class_labels)):
    plt.plot(fpr[i], tpr[i], label=f'{class_labels[i]} (AUC = {roc_auc[i]:.4f})') 

plt.xlim(zoom_x_start, zoom_x_end)
plt.ylim(zoom_y_start, zoom_y_end)
plt.legend(loc='lower right')
plt.tight_layout() 
plt.savefig(path+'roc_curves_zoomed.pdf', format='pdf')
plt.show()
plt.close()

plt.figure(figsize=(5, 4))
sns.heatmap(cm / cm.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2%', cmap='Blues', xticklabels=class_labels.values(), yticklabels=class_labels.values())

tick_labels = [f"{class_labels[i]} ({i})" for i in range(len(class_labels))]
plt.xticks(np.arange(len(class_labels)) + 0.5, tick_labels, rotation=45, ha='right')
plt.yticks(np.arange(len(class_labels)) + 0.5, tick_labels, rotation=0)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(path+"confusion_matrix.pdf", format="pdf", bbox_inches='tight')
plt.show()
