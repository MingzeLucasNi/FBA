'''
Adversarial example is defined as input with maliciously imperciptible noise to deprave the quality of the model, therefore we test attacking methods in terms of imperciptiness and attacking performance. To be more specific, we test imperciptiness by comparing the semantic and syntactic imperciptibility,which will be measured by USE and Rouge, bewteen the input and the original sentence and its adversarial example. As for attacking performance, we test the performance of the model by the logits drop (LOD) of the model on the original sentence and its adversarial example, and the attacking success rate(ASR).
'''
from rouge_score import rouge_scorer
import transformers
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from transformers import AutoTokenizer, AutoModel,PretrainedConfig,AutoConfig #config for the 

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
def Rouge(input,ref):
    '''
    input: the input sentence
    ref: the reference sentence
    '''
    scores = scorer.score(input, ref)['rouge1'].precision
    return scores



# Since sentence similarity is a very difficult task, we use the MiniMl-L6 to make sentences embeddings and make cosine similarity to measure the similarity between the input and the reference sentence. 

Mini_model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') #import the Sentence embedding model
Cos=nn.CosineSimilarity(dim=0, eps=1e-6) #define the cosine similarity function

def sentence_similarity(input,ref):
    '''
    input: the input sentence
    ref: the reference sentence
    '''
    example=[input,ref]
    embeddings=Mini_model.encode(example)
    similarity=Cos(torch.from_numpy(embeddings[0]), torch.from_numpy(embeddings[1]))
    return similarity






# Besides the Metrics used for training, we also imply Universal Sentence Encoder(USE) for evaluating semantic imperciptibility. There are several versions of USE, and we take the USE-v4 as it is the most popular one.

# module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
# USE_model = hub.load(module_url)

# def USE(input,ref):
#     encoding=USE_model([input,ref])
#     sim=np.inner(encoding[0],encoding[1])
#     return sim
