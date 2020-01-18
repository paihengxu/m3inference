from m3inference.text_model import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from m3inference.dataset import *

# 10 in the paper
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using", device)
num_epoch = 5
m3text = M3InferenceTextModel()
m3text.to(device)
# TODO: change the dataset
data_or_datapath = "test/test_data.jsonl"

data = []
with open(data_or_datapath) as f:
    for line in f:
        data.append(json.loads(line))

# Loading dataset
dataloader = DataLoader(M3InferenceDataset(data, use_img=False), 16,
                        num_workers=4, pin_memory=True)
criterion = nn.CrossEntropyLoss()
# learning rate 0.001 for ground-up model, 0.0005 for fine-tuning model
optimizer = torch.optim.Adam(m3text.parameters(), lr=0.001, amsgrad=True)
# y_pred = []
for epoch in range(num_epoch):
    epoch_loss = 0
    acc_num = 0
    num = 0
    # for batch in tqdm(dataloader, desc='Training...'):
    for batch in dataloader:
        # print(batch)
        data = [i.to(device) for i in batch['data']]
        # extract the label from batch
        # label = [i.to(device) for i in batch['label']]
        label = batch['label'].to(device)
        # pred = self.model(batch)
        output = m3text.forward(data, 'gender')
        # print(output)
        pred = torch.argmax(output, dim=1)
        # print("pred", pred)
        # print("label", label)
        acc = sum((pred==label).cpu().detach().numpy())
        # print(acc)
        acc_num += acc
        num += len(label)
        # y_pred.append([_pred.detach().cpu().numpy() for _pred in pred])
        loss = criterion(output, label)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    print("epoch: {}, loss: {}, acc: {}, num :{}".format(epoch+1, epoch_loss, acc_num/num, num))

torch.save(m3text.state_dict(), 'text_test.mdl')
