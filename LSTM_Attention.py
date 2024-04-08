import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import cv2
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(self.hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: [batch_size, seq_length, hidden_size]
        attention_weights = torch.softmax(self.attention(lstm_output).squeeze(2), dim=1)
        # attention_weights shape: [batch_size, seq_length]
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        # context_vector shape: [batch_size, hidden_size]
        return context_vector, attention_weights

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        context_vector, attention_weights = self.attention(lstm_out)
        out = self.fc(context_vector)
        return out

class JointDataset(Dataset):
    def __init__(self, joints_paths, labels):
        self.joints_paths = joints_paths
        self.labels = labels
        
    def __len__(self):
        return len(self.joints_paths)
    
    def __getitem__(self, idx):
        joints_path = self.joints_paths[idx]
        label = self.labels[idx]
        joints = torch.load(joints_path)
        joints = joints.view(joints.size(0), -1)
        return joints, label

def labels_frequency(labels):
    labels_freq={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0}
    count=0.0
    for label in labels:
        labels_freq[int(label)]+=1
        count+=1
    for key in labels_freq:
        labels_freq[key]=labels_freq[key]/count
    return labels_freq

def weighted_loss(labels):
    labels_freq=labels_frequency(labels)
    weights=[1/labels_freq[i] for i in range(8)]
    return weights

def train_LSTM(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, args, device):
    model.to(device)
    epoch_loss, lrs = {"train": [], "val": []}, []
    for epoch in tqdm(range(args.epochs)):
        batch_loss = []
        model.train()
        for features, labels in tqdm(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            #forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)   
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        cur_epoch_loss = np.mean(batch_loss)
        epoch_loss["train"].append(cur_epoch_loss)
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss = {cur_epoch_loss}")
        
        # check the eval loss
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            val_batch_loss = []
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = criterion(model(features), labels) 
                val_batch_loss.append(loss.item())

        #get the accuracy on the validation set
        with torch.no_grad():
            correct = 0
            total = 0
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Val Accuracy: {100 * correct / total}")
        
        #get f1 score
        y_true = []
        y_pred = []
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"F1 score on val: {100*f1}")
        epoch_loss["val"].append(np.mean(val_batch_loss))
        scheduler.step()

        lrs.append(scheduler.get_last_lr())

    #save the model
    torch.save(model.state_dict(), f"{args.output}/trained_model.pth")

    #evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Final model accuracy: {100 * correct / total}")

    return model, epoch_loss, lrs

def build_joint_paths_and_labels(examples_dir_path, annotations_json_path):
    joints_paths = []
    labels = []
    examples_dir = os.listdir(examples_dir_path)
    annotations_json_path = json.load(open(annotations_json_path))
    for example in examples_dir:
        if example.endswith(".pt") and example[:-3] in annotations_json_path:
            if annotations_json_path[example[:-3]] == 9 or annotations_json_path[example[:-3]] == 8:
                if np.random.binomial(1, 0.3):
                    joints_paths.append(os.path.join(examples_dir_path, example))
                    labels.append(7)
            else:
                joints_paths.append(os.path.join(examples_dir_path, example))
                labels.append(annotations_json_path[example[:-3]])


    return joints_paths, labels

def collate_fn(batch):
    joints, labels = zip(*batch)
    joints = torch.nn.utils.rnn.pad_sequence(joints, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return joints, labels

def get_loaders(args):

    joints_paths, labels = build_joint_paths_and_labels("skeletal_dataset2", "dataset/annotation_dict.json")
    X_train, X_test, y_train, y_test = train_test_split(joints_paths, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    train_dataset = JointDataset(X_train, y_train)
    val_dataset = JointDataset(X_val, y_val)
    test_dataset = JointDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def train_classifier(args):
    # freeze seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.no_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, test_loader = get_loaders(args)
    
    model = LSTMWithAttention(args.input_size, 126, args.num_layers, 8, device)


    _, labels = build_joint_paths_and_labels("skeletal_dataset2", "dataset/annotation_dict.json")
    #weighted loss
    w_loss = torch.tensor(weighted_loss(labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=w_loss)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # create cosine lr scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    model, epoch_loss, lrs = train_LSTM(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, args, device)

