import pandas as pd
from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

#helper classes
class DatasetInfo:
    def __init__(self, name, uci_id, columns_to_drop):
        self.name = name                      
        self.uci_id = uci_id                  
        self.columns_to_drop = columns_to_drop

class PreparedData:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, category_names, num_features, num_classes):
        self.X_train = X_train       
        self.X_val = X_val           
        self.X_test = X_test  
        self.y_train = y_train      
        self.y_val = y_val           
        self.y_test = y_test         
        self.category_names = category_names 
        self.num_features = num_features     
        self.num_classes = num_classes       



def load_dataset(dataset_info):
    dataset = fetch_ucirepo(id=dataset_info.uci_id)
    features = pd.DataFrame(dataset.data.features)

    labels = pd.DataFrame(dataset.data.targets).iloc[:, 0]

    for col in dataset_info.columns_to_drop:
            if col in features.columns: 
                features = features.drop(columns=[col])
            
    labels_clean = labels.dropna()
    features_clean = features.loc[labels_clean.index]
    
    return features_clean.reset_index(drop=True), labels_clean.reset_index(drop=True)



#data preparation function
def prepare_and_split_data(features, labels):
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels.astype(str))
    class_names = list(encoder.classes_)

    features_numeric = features.apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    features_imputed = imputer.fit_transform(features_numeric)

    X_train, X_temp, y_train, y_temp = train_test_split(features_imputed, labels_encoded, test_size=0.40, random_state=42, stratify=labels_encoded)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    data = PreparedData(
        X_train=X_train_scaled, 
        X_val=X_val_scaled, 
        X_test=X_test_scaled, 
        y_train=y_train, 
        y_val=y_val, 
        y_test=y_test, 
        category_names=class_names, 
        num_features=X_train_scaled.shape[1], 
        num_classes=len(class_names)
    )
    
    return data