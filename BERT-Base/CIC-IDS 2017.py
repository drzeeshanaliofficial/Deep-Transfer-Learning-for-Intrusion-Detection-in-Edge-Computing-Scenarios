import numpy as np 
import pandas as pd
from google.colab import drive
import pandas as pd
import numpy as np

drive.mount('/content/drive')
path = "/content/drive/My Drive/CIC-IDS 2017 Dataset/"

df1=pd.read_csv(path+"Monday-WorkingHours.pcap_ISCX.csv")
df2=pd.read_csv(path+"Tuesday-WorkingHours.pcap_ISCX.csv")
df3=pd.read_csv(path+"Wednesday-workingHours.pcap_ISCX.csv")
df4=pd.read_csv(path+"Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
df5=pd.read_csv(path+"Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
df6=pd.read_csv(path+"Friday-WorkingHours-Morning.pcap_ISCX.csv")
df7=pd.read_csv(path+"Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df8=pd.read_csv(path+"Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

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

label_mapping = {
    'BENIGN': 'Normal',
    'Web Attack � Brute Force': 'Web Attacks',
    'Web Attack � XSS': 'Web Attacks',
    'Web Attack � Sql Injection': 'Web Attacks',
}

df[' Label']=df[' Label'].replace(label_mapping)

duplicate_samples = df[df.duplicated()]
print("Duplicate Samples Count:", len(duplicate_samples))
duplicate_samples_per_class = duplicate_samples[' Label'].value_counts()
print("Duplicate Samples Count per Class:")
print(duplicate_samples_per_class)
df = df.drop_duplicates()

duplicate_samples_after_removal = df[df.duplicated()]

label_mapping = {
    'FTP-Patator':'Brute Force',
    'SSH-Patator':'Brute Force',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'DoS Hulk': 'DoS',
    'Heartbleed':'DoS',
    'Bot':'DoS',
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
    "Normal": 500000                    
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
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
X_train_features = []
for i in tqdm(range(0, len(X_train_input_ids), 100)):
    batch_input_ids = X_train_input_ids[i:i+100]
    batch_attention_mask = X_train_attention_mask[i:i+100]
    features = bert_model([batch_input_ids, batch_attention_mask])[0][:, 0, :].numpy()
    X_train_features.append(features)
X_train_features = np.concatenate(X_train_features, axis=0)
print("Train features shape:", X_train_features.shape)

np.save(path+'train_features_BERT_Base.npy', X_train_features)

#Testing Set Tokenization & Feature Extraction
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
from tqdm import tqdm

flow_text_test = [' '.join(map(str, row)) for row in X_test]
print("Number of Rows in flow_text_test:", len(flow_text_test))

X_test_tokenized = []
print("\nTokenizing the Testing Set:")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
X_test_features = []
for i in tqdm(range(0, len(X_test_input_ids), 100)):
    batch_input_ids = X_test_input_ids[i:i+100]
    batch_attention_mask = X_test_attention_mask[i:i+100]
    features = bert_model([batch_input_ids, batch_attention_mask])[0][:, 0, :].numpy()
    X_test_features.append(features)
X_test_features = np.concatenate(X_test_features, axis=0)
print("Test features shape:", X_test_features.shape)

np.save(path+'test_features_BERT_Base.npy', X_test_features)

print(X_train.shape)
print(y_train.shape)
print()
print(X_test.shape)
print(y_test.shape)
print()

train_features = np.load(path+'train_features_BERT_Base.npy')
print(train_features.shape)

test_features = np.load(path+'test_features_BERT_Base.npy')
print(test_features.shape)

# MLP Training
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold

train_features = np.load(path+'train_features_BERT_Base.npy')
train_labels = y_train

test_features = np.load(path+'test_features_BERT_Base.npy')
test_labels = y_test

label_encoder = LabelEncoder()
label_encoder.fit(train_labels)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

mlp = MLPClassifier(hidden_layer_sizes=(256,), activation='logistic', random_state=42, max_iter=200, learning_rate='adaptive')
num_epochs = 1
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []

for epoch in range(num_epochs):
    fold_accuracy_scores = []
    fold_precision_scores = []
    fold_recall_scores = []
    fold_f1_scores = []
    fold_roc_auc_scores = []

    for train_index, val_index in cv.split(train_features, train_labels):
        X_train, X_val = train_features[train_index], train_features[val_index]
        y_train, y_val = train_labels[train_index], train_labels[val_index]
        mlp.partial_fit(X_train, y_train, classes=label_encoder.classes_)
        predictions = mlp.predict(X_val)
        accuracy = accuracy_score(label_encoder.transform(y_val), label_encoder.transform(predictions))
        precision = precision_score(label_encoder.transform(y_val), label_encoder.transform(predictions), average='weighted')
        recall = recall_score(label_encoder.transform(y_val), label_encoder.transform(predictions), average='weighted')
        f1 = f1_score(label_encoder.transform(y_val), label_encoder.transform(predictions), average='weighted')
        roc_auc = roc_auc_score(label_encoder.transform(y_val), mlp.predict_proba(X_val), multi_class='ovr')
        fold_accuracy_scores.append(accuracy)
        fold_precision_scores.append(precision)
        fold_recall_scores.append(recall)
        fold_f1_scores.append(f1)
        fold_roc_auc_scores.append(roc_auc)
    accuracy_scores.append(np.mean(fold_accuracy_scores))
    precision_scores.append(np.mean(fold_precision_scores))
    recall_scores.append(np.mean(fold_recall_scores))
    f1_scores.append(np.mean(fold_f1_scores))
    roc_auc_scores.append(np.mean(fold_roc_auc_scores))

    print("Epoch:", epoch+1)
    print("Accuracy:", accuracy_scores[-1])
    print("Precision:", precision_scores[-1])
    print("Recall:", recall_scores[-1])
    print("F1-Score:", f1_scores[-1])
    print("ROC AUC:", roc_auc_scores[-1])
    print()

mlp.fit(train_features, train_labels)
test_labels_encoded = label_encoder.transform(test_labels)
predictions = mlp.predict(test_features)
predictions_encoded = label_encoder.transform(predictions)

accuracy = accuracy_score(test_labels_encoded, predictions_encoded)
precision = precision_score(test_labels_encoded, predictions_encoded, average='weighted')
recall = recall_score(test_labels_encoded, predictions_encoded, average='weighted')
f1 = f1_score(test_labels_encoded, predictions_encoded, average='weighted')
roc_auc = roc_auc_score(test_labels_encoded, mlp.predict_proba(test_features), multi_class='ovr')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC AUC:", roc_auc)

test_labels_str = label_encoder.inverse_transform(test_labels_encoded)
predictions_str = label_encoder.inverse_transform(predictions_encoded)

cm = confusion_matrix(test_labels_str, predictions_str)
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

joblib.dump(mlp, path+'mlp_classifier.pkl')

import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, precision_recall_curve, classification_report

train_features = np.load(path+'train_features_BERT_Base.npy')
train_labels = y_train

test_features = np.load(path+'test_features_BERT_Base.npy')
test_labels = y_test

loaded_mlp = joblib.load(path+'mlp_classifier.pkl')

label_encoder = LabelEncoder()
label_encoder.fit(train_labels)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

predictions_loaded = loaded_mlp.predict(test_features)
test_labels_encoded = label_encoder.transform(test_labels)

predictions_encoded = label_encoder.transform(predictions_loaded)
test_labels_str = label_encoder.inverse_transform(test_labels_encoded)
predictions_str = label_encoder.inverse_transform(predictions_encoded)
predictions_loaded_encoded = label_encoder.transform(predictions_loaded)
accuracy_loaded = accuracy_score(test_labels_encoded, predictions_loaded_encoded)
precision_loaded = precision_score(test_labels_encoded, predictions_loaded_encoded, average='weighted')
recall_loaded = recall_score(test_labels_encoded, predictions_loaded_encoded, average='weighted')
f1_loaded = f1_score(test_labels_encoded, predictions_loaded_encoded, average='weighted')
roc_auc_loaded = roc_auc_score(test_labels_encoded, loaded_mlp.predict_proba(test_features), multi_class='ovr')
print("Accuracy (Loaded Model):", accuracy_loaded)
print("Precision (Loaded Model):", precision_loaded)
print("Recall (Loaded Model):", recall_loaded)
print("F1-Score (Loaded Model):", f1_loaded)
print("ROC AUC (Loaded Model):", roc_auc_loaded)

cm_loaded = confusion_matrix(test_labels_str, label_encoder.inverse_transform(predictions_loaded_encoded))

cm_loaded_percent = cm_loaded.astype('float') / cm_loaded.sum(axis=1)[:, np.newaxis]
class_labels = label_encoder.classes_
plt.figure(figsize=(8, 5)) 
sns.heatmap(cm_loaded_percent, annot=True, cmap='Blues', fmt='.2%')  
tick_labels = [f"{class_labels[i]} ({i})" for i in range(len(class_labels))]
plt.xticks(np.arange(len(class_labels)) + 0.5, tick_labels, rotation=45, ha='right')
plt.yticks(np.arange(len(class_labels)) + 0.5, tick_labels, rotation=0)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

classification_report_str = classification_report(test_labels_encoded, predictions_loaded_encoded, target_names=class_labels, digits=4)
print("Classification Report (Loaded Model):\n")
print(classification_report_str)

class_accuracies = {}
for i, label in enumerate(class_labels):
    true_mask = test_labels_encoded == i
    correct_predictions = np.sum(test_labels_encoded[true_mask] == predictions_loaded_encoded[true_mask])
    total_samples = np.sum(true_mask)
    class_accuracy = correct_predictions / total_samples
    class_accuracies[label] = class_accuracy

print("\nClass-wise Accuracies:")
for label, accuracy in class_accuracies.items():
    print(f"Class: {label}, Accuracy: {accuracy:.2f}")


import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, precision_recall_curve, classification_report

train_features = np.load(path+'train_features_BERT_Base.npy')
train_labels = y_train

test_features = np.load(path+'test_features_BERT_Base.npy')
test_labels = y_test

loaded_mlp = joblib.load(path+'mlp_classifier.pkl')

label_encoder = LabelEncoder()
label_encoder.fit(train_labels)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

predictions_loaded = loaded_mlp.predict(test_features)
test_labels_encoded = label_encoder.transform(test_labels)

predictions_encoded = label_encoder.transform(predictions_loaded)

test_labels_str = label_encoder.inverse_transform(test_labels_encoded)
predictions_str = label_encoder.inverse_transform(predictions_encoded)

predictions_loaded_encoded = label_encoder.transform(predictions_loaded)
accuracy_loaded = accuracy_score(test_labels_encoded, predictions_loaded_encoded)
precision_loaded = precision_score(test_labels_encoded, predictions_loaded_encoded, average='weighted')
recall_loaded = recall_score(test_labels_encoded, predictions_loaded_encoded, average='weighted')
f1_loaded = f1_score(test_labels_encoded, predictions_loaded_encoded, average='weighted')
roc_auc_loaded = roc_auc_score(test_labels_encoded, loaded_mlp.predict_proba(test_features), multi_class='ovr')

print("Accuracy (Loaded Model):", accuracy_loaded)
print("Precision (Loaded Model):", precision_loaded)
print("Recall (Loaded Model):", recall_loaded)
print("F1-Score (Loaded Model):", f1_loaded)
print("ROC AUC (Loaded Model):", roc_auc_loaded)
cm_loaded = confusion_matrix(test_labels_str, label_encoder.inverse_transform(predictions_loaded_encoded))
cm_loaded_percent = cm_loaded.astype('float') / cm_loaded.sum(axis=1)[:, np.newaxis]

class_labels = label_encoder.classes_
plt.figure(figsize=(8, 5)) 
annot = np.empty_like(cm_loaded).astype(str)
for i in range(cm_loaded.shape[0]):
    for j in range(cm_loaded.shape[1]):
        annot[i, j] = f"{cm_loaded_percent[i, j]:.2%}\n({cm_loaded[i, j]})"

sns.heatmap(cm_loaded_percent, annot=annot, cmap='Blues', fmt='') 

tick_labels = [f"{class_labels[i]} ({i})" for i in range(len(class_labels))]
plt.xticks(np.arange(len(class_labels)) + 0.5, tick_labels, rotation=45, ha='right')
plt.yticks(np.arange(len(class_labels)) + 0.5, tick_labels, rotation=0)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

classification_report_str = classification_report(test_labels_encoded, predictions_loaded_encoded, target_names=class_labels, digits=4)
print("Classification Report (Loaded Model):\n")
print(classification_report_str)

class_accuracies = {}
for i, label in enumerate(class_labels):
    true_mask = test_labels_encoded == i
    correct_predictions = np.sum(test_labels_encoded[true_mask] == predictions_loaded_encoded[true_mask])
    total_samples = np.sum(true_mask)
    class_accuracy = correct_predictions / total_samples
    class_accuracies[label] = class_accuracy

print("\nClass-wise Accuracies:")
for label, accuracy in class_accuracies.items():
    print(f"Class: {label}, Accuracy: {accuracy:.2f}")

annot = np.empty_like(cm_loaded).astype(str)
for i in range(cm_loaded.shape[0]):
    for j in range(cm_loaded.shape[1]):
        annot[i, j] = f"{cm_loaded_percent[i, j]:.2%}\n({cm_loaded[i, j]})"

sns.heatmap(cm_loaded_percent, annot=annot, cmap='Blues', fmt='', cbar=False) 

tick_labels = [f"{class_labels[i]} ({i})" for i in range(len(class_labels))]
plt.xticks(np.arange(len(class_labels)) + 0.5, tick_labels, rotation=45, ha='right')
plt.yticks(np.arange(len(class_labels)) + 0.5, tick_labels, rotation=0)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(path+"confusion_matrix.pdf", format="pdf", bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 5))  
sns.heatmap(cm_loaded_percent, annot=True, cmap='Blues', fmt='.2%') 
tick_labels = [f"{class_labels[i]} ({i})" for i in range(len(class_labels))]
plt.xticks(np.arange(len(class_labels)) + 0.5, tick_labels, rotation=45, ha='right')
plt.yticks(np.arange(len(class_labels)) + 0.5, tick_labels, rotation=0)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.savefig(path+"confusion_matrix.pdf", format="pdf", bbox_inches='tight')
plt.show()

fpr = {}
tpr = {}
roc_auc = {}

for i, label in enumerate(class_labels):
    fpr[i], tpr[i], _ = roc_curve(test_labels_encoded == i, loaded_mlp.predict_proba(test_features)[:, i])
    roc_auc[i] = roc_auc_score(test_labels_encoded == i, loaded_mlp.predict_proba(test_features)[:, i])

plt.figure(figsize=(5, 4))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

zoom_x_start = 0.0  
zoom_x_end = 1.0    
zoom_y_start = 0.995 
zoom_y_end = 1.0    

for i, label in enumerate(class_labels):
    plt.plot(fpr[i], tpr[i], label=f'{label} (AUC = {roc_auc[i]:.4f})')

plt.xlim(zoom_x_start, zoom_x_end)
plt.ylim(zoom_y_start, zoom_y_end)
plt.legend(loc='lower right')

plt.tight_layout() 
plt.savefig(path+'roc_curves_multiclass_zoomed.pdf', format='pdf')
plt.show()
plt.close()

class_probabilities = loaded_mlp.predict_proba(test_features)

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision = dict()
recall = dict()
thresholds = dict()

for i in range(len(class_labels)):
    precision[i], recall[i], thresholds[i] = precision_recall_curve(
        test_labels_encoded == i, class_probabilities[:, i])

plt.figure(figsize=(5, 4))  
plt.xlabel('Recall')
plt.ylabel('Precision')

zoom_x_start = 0.96  
zoom_x_end = 1.0   
zoom_y_start = 0.96  
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
