'''
In this file, we tend to show the utility functions, inlcuding:
    1. victim classifiers for different datasets:
        a. AG NEWS: textattack/bert-base-uncased-ag-news
        b. emotion: bhadresh-savani/distilbert-base-uncased-emotion
        b. SST-2: textattack/roberta-base-SST-2 
'''
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForMaskedLM
from sklearn.neighbors import NearestNeighbors
import torch

from torch.nn import functional as F
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torch

class rouge_score:
    def __init__(self,rouge_type):
        '''
        Usage: to calculate the different rouge score, usually we use 'rouge1'
        '''
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    def Rouge(self,input,ref):
        '''
        input: the input sentence
        ref: the reference sentence
        '''
        scores = self.scorer.score(input, ref)['rouge1'].precision
        return scores


class sentence_similarity:
    '''
    Usage: to calculate the sentence similarity
    '''
    def __init__(self, name='sentence-transformers/all-MiniLM-L6-v2'):
        self.Mini_model=SentenceTransformer(name) #import the Sentence embedding model
        self.Cos=nn.CosineSimilarity(dim=0, eps=1e-6) #define the cosine similarity function

    def sem(self,input,ref):
        '''
        input: the input sentence
        ref: the reference sentence
        '''
        example=[input,ref]
        embeddings=self.Mini_model.encode(example)
        similarity=self.Cos(torch.from_numpy(embeddings[0]), torch.from_numpy(embeddings[1]))
        return similarity


class embed_prob:
    def __init__(self):
        embedding_matrix=torch.load('with_embed.pt')
        self.neighbors=NearestNeighbors(n_neighbors=5)
        self.neighbors.fit(embedding_matrix)
        # self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.model = AutoModelForMaskedLM.from_pretrained("roberta-large", output_hidden_states=True)
    def prob(self,id):
        input=torch.tensor([id])
        input=torch.cat((torch.tensor([0]),input,torch.tensor([2]))).unsqueeze(0)
        can=self.model(input)['hidden_states'][0].squeeze()[1,:].tolist()
        can=[can]
        indices=self.neighbors.kneighbors(can, return_distance=False)
        indices=torch.from_numpy(indices.squeeze())[1:]
        return indices




class victim_models:
    def __init__(self,model_name):
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def logits(self,text,label):
        inputs=self.tokenizer.encode(text, return_tensors="pt")
        outputs=self.model(inputs).logits.squeeze()
        logits=outputs[label]
        return logits

    def prob(self,text,label):
        inputs=self.tokenizer.encode(text, return_tensors="pt")
        outputs=self.model(inputs).logits.squeeze()
        prob=F.softmax(outputs)[label]
        return prob

    def predict(self,text):
        inputs=self.tokenizer.encode(text, return_tensors="pt")
        outputs=self.model(inputs).logits.squeeze()
        prob=F.softmax(outputs).squeeze()
        pre_label=torch.argmax(prob)
        return pre_label

