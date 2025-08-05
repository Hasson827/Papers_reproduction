# 将所有的训练过程模块化，集中在utils.py文件中

#1. 导入必要的库
import numpy as np
import torch 
import matplotlib.pyplot as plt
from torch import nn,optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from PIL import Image



#2. 定义最基本的分类训练模型
def train_cluster(train_iter,model,loss_fn,optimizer,device):
    model = model.to(device)
    
    model.train()
    running_loss = 0
    running_acc = 0
    
    for X,y in train_iter:
        X = X.to(device)
        y = y.to(device)
        
        y_pred_logit = model(X)
        y_pred_label = torch.argmax(torch.softmax(y_pred_logit,dim=1),dim=1)

        loss = loss_fn(y_pred_logit,y)
        running_loss += loss.item()*X.shape[0]
        running_acc += (y_pred_label == y).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_iter.dataset)
    epoch_acc = running_acc*100 / len(train_iter.dataset)
    
    return epoch_loss, epoch_acc, model, optimizer



#3. 定义最基本的分类测试模型
def test_cluster(test_iter,model,loss_fn,device):
    model = model.to(device)
    
    model.eval()
    running_loss = 0
    running_acc = 0
    
    for X,y in test_iter:
        X = X.to(device)
        y = y.to(device)
        
        y_pred_logit = model(X)
        loss = loss_fn(y_pred_logit,y)
        running_loss += loss.item()*X.shape[0]
        y_pred_label = torch.argmax(torch.softmax(y_pred_logit,dim=1),dim=1)
        running_acc += (y_pred_label == y).sum().item()
    
    epoch_loss = running_loss / len(test_iter.dataset)
    epoch_acc = running_acc*100 / len(test_iter.dataset)
    
    return epoch_loss, epoch_acc, model



#4. 定义画图函数
def plot_loss_acc(train_loss,test_loss,train_acc,test_acc):
    
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)
    
    fig, ax1 = plt.subplots(figsize = (8,4.5))
    
    ax1.plot(train_loss, linestyle = '--', color = 'blue', label = 'Training Loss')
    ax1.plot(test_loss, linestyle = '--', color = 'red', label = 'Testing Loss')
    ax1.set_ylabel('Loss')
    
    ax2 = ax1.twinx()
    ax2.plot(train_acc, linestyle = '-', color = 'blue', label = 'Training Accuracy/%')
    ax2.plot(test_acc, linestyle = '-', color = 'red', label = 'Testing Accuracy/%')
    ax2.set_ylabel('Accuracy/%')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.show()



#5. 定义训练+测试循环
def training_loop(model,loss_fn,optimizer,train_iter,test_iter,epochs,device,print_every=1):
    print(f"Training and Testing on {device}")
    train_losses,test_losses = [],[]
    train_accs,test_accs = [],[]    
    
    for epoch in range(epochs):
        train_loss, train_acc, model, optimizer = train_cluster(train_iter,model,loss_fn,optimizer,device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        with torch.inference_mode():
            test_loss, test_acc, model = test_cluster(test_iter,model,loss_fn,device)
            
            test_losses.append(test_loss)
            test_accs.append(test_acc)
        
        if epoch % print_every == 0:
            print(f"Epoch{epoch+1} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%\n"
                  f"       | Test Loss:  {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
        
    plot_loss_acc(train_losses,test_losses,train_accs,test_accs)