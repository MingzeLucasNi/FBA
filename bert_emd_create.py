from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import time
tokenizer = AutoTokenizer.from_pretrained("roberta-large")

model = AutoModelForMaskedLM.from_pretrained("roberta-large", output_hidden_states=True)

def hidden_features(model, id):
    # encoded_input = tokenizer.encode(text, return_tensors='pt')
    encoded_input = torch.tensor([id])
    start=torch.tensor([0])
    end=torch.tensor([2])
    encoded_input = torch.cat((start, encoded_input, end)).unsqueeze(0)
    # print(encoded_input.shape)
    encoded_input_no = encoded_input[:, 1:-1]
    encoded_input_yes = encoded_input
    hidden_states_no = model(encoded_input_no)[1][0].squeeze()
    hidden_states_yes = model(encoded_input_yes)[1][0].squeeze()[1,:]
    return hidden_states_no, hidden_states_yes

# hidden_features(model, 1230)[1].shape

with_special_embeddings=[]
without_special_embeddings=[]
begin=time.time()
for i in range(len(tokenizer)):
    without_emd,with_emd=hidden_features(model, i)
    with_special_embeddings.append(with_emd)
    without_special_embeddings.append(without_emd)
    finish=time.time()
    print('############  {}/{}  ##########'.format(i,len(tokenizer)))
    print('############ remaining time:{} ###########'.format((finish-begin)/60/(i+1)*(len(tokenizer)-i)))
torch.save(with_special_embeddings, 'with_embed.pt')
torch.save(without_special_embeddings, 'without_special_embeddings.pt')
