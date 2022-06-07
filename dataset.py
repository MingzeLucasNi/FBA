##import packages
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM




# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class MHDataset(Dataset):
    def __init__(self, data_name, tokenizer,cuda=False):
        self.data = torch.load('datasets/'+data_name+'.pt')['test']
        self.cuda = cuda
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data['feature'])

    def __getitem__(self, idx):
        tokens=self.tokenizer(self.data['feature'],truncation=True, padding=True,return_tensors='pt')['input_ids']
        text_id=self.tokenizer.decode(tokens[idx],skip_special_tokens=True)
        # print(type(text_id))
        input_id=self.tokenizer.encode(text_id,return_tensors='pt').squeeze()
        # print('the original input is:',input_id.shape)
        label_id=self.data['label'][idx]
        return text_id,input_id, label_id
# s=[]

# d=MHDataset('ag_news',BertTokenizer.from_pretrained('roberta-large'))
# d_loader=DataLoader(d,batch_size=1,shuffle=False)
# for i,(text_id,input_id,label_id) in enumerate(d_loader):
#     # print(text_id,input_id,label_id)
#     # print(type(text_id))
#     # print(type(input_id))
#     # print(type(label_id))
#     # print(input_id.shape)
#     s.append(input_id.shape[0])
#     # print(label_id.shape)
#     # print(label_id.shape)
#     # print(text_id[0])
#     # print(input_id.squeeze())

    