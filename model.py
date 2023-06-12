import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as opt
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.rcParams['font.sans-serif'] = ['Times New Roman']
import DataAnalysis


# 模型构建
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,16)
        self.linear3 = nn.Linear(16,8)
        self.linear4 = nn.Linear(8,4)
        self.linear5 = nn.Linear(4,1)

        self.act = nn.Sigmoid()

    def forward(self, x):

        output_1 = self.act(self.linear1(x))
        output_2 = self.act(self.linear2(output_1))
        output_2 = self.act(self.linear3(output_2))
        output_2 = self.act(self.linear4(output_2))
        output_3 = self.act(self.linear5(output_2))

        return output_3.squeeze(1)


# tensor转换,这里y_train是一个Series需要调用values
x_train = torch.tensor(DataAnalysis.x_train,dtype=torch.float32)
y_train = torch.tensor(DataAnalysis.y_train.values,dtype=torch.float32)

x_test = torch.tensor(DataAnalysis.x_test,dtype=torch.float32)
y_test = torch.tensor(DataAnalysis.y_test.values,dtype=torch.float32)
#print(x_train.shape) (712,56)
#成对
train_load = TensorDataset(x_train, y_train)
test_load = TensorDataset(x_test, y_test)
#数据加载
load_train = DataLoader(train_load, batch_size=64, shuffle=True)
load_test = DataLoader(test_load, batch_size=64, shuffle=False)
# 模型训练


base_net = DNN(input_size=56,hidden_size=32,output_size=1)
criterion = nn.BCELoss()
optimizer = opt.Adam(params=base_net.parameters(),lr=0.01)
Epoch = 20
base_net.train()
for epoch in range(Epoch):

    for batch_ndx, sample in enumerate(load_train):
        base_net.zero_grad()
        x_data, label = sample
        predict_value = base_net(x_data)
        loss = criterion(predict_value,label)
        loss.backward()
        optimizer.step()
    #epoch_precision = epoch_precision
    print("Epoch:{:04d},".format(epoch+1))

base_net.eval()
with torch.no_grad():
    num = 0
    epoch_precision = 0.0

    for batch_ndx, sample in enumerate(load_test):
        x_data, label = sample
        predict_value = base_net(x_data)
        if num == 0:
            all_predict_value = predict_value
            all_y_true = label
        else:
            all_predict_value = torch.cat([all_predict_value,predict_value])
            all_y_true = torch.cat([all_y_true, label])
        loss = criterion(predict_value, label)
        num += 1
    #可设置阈值判别0或者1，目前最好的是1recall，0.91的precision
    torch.where(torch.lt(all_predict_value,0.91),0,all_predict_value)
    cm = confusion_matrix(all_y_true, all_predict_value.round())
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("precision:{:02.4f},recall:{:02.4f}".format(precision,recall))

