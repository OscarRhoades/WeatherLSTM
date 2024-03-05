import preprocessing
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



# preprocessed_data = preprocessing.preprocess_synthetic('weatherAUS.csv')

# preprocessed_data.to_csv('synthetic_data.csv', index=False)



class CustomDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

def split_data(preprocessed_data):
    
    # preprocessed_data = preprocessing.preprocess_synthetic('weatherAUS.csv')
    # Split data into test and target
    features = preprocessed_data.drop('RainTomorrow', axis=1).values
    target = preprocessed_data['RainTomorrow'].values

    # typecast
    features_tensor = torch.tensor(features, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.float32)

    # split
    X_train, X_test, y_train, y_test = train_test_split(features_tensor, target_tensor, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    batch_size = 32  # Set your desired batch size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
