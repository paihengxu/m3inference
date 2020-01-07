from m3inference.text_model import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from m3inference.dataset import *

# 10 in the paper
device = 'cpu'
num_epoch = 1
m3text = M3InferenceTextModel()
data_or_datapath = "test/data.jsonl"

data = []
with open(data_or_datapath) as f:
    for line in f:
        data.append(json.loads(line))

# Loading dataset
dataloader = DataLoader(M3InferenceDataset(data, use_img=False), 1,
                        num_workers=4, pin_memory=True)
criterion = nn.CrossEntropyLoss()
# learning rate 0.001 for ground-up model, 0.0005 for fine-tuning model
optimizer = torch.optim.Adam(m3text.parameters(), lr=0.001, amsgrad=True)
y_pred = []
for epoch in range(num_epoch):
    epoch_loss = 0
    for batch in tqdm(dataloader, desc='Training...'):
        print(batch)
        # batch = [i.to(device) for i in batch]
        data = batch['data']
        # extract the label from batch
        # label = [i.to(device) for i in batch['label']]
        label = batch['label']
        # pred = self.model(batch)
        output = m3text.forward(data, 'gender')
        print(output)
        # y_pred.append([_pred.detach().cpu().numpy() for _pred in pred])
        loss = criterion(output, label)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

