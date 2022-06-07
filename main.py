from tqdm import tqdm
from mcmc import *
from uti import *
from dataset import *
import torch
import torch.nn as nn

import numpy as np
import argparse
import torch.utils.data as DataLoader
import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM



def sampling_attacker(args):

    # BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # BERT = BertForMaskedLM.from_pretrained('bert-base-uncased')
    BERT_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    BERT = AutoModelForMaskedLM.from_pretrained("roberta-large")
    semantic_sim = sentence_similarity()
    rouge_sim=rouge_score('rouge1')
    victim_classifier =victim_models(args.victim_model_name)
    embed_search=embed_prob()
    dataset=MHDataset(args.dataset, BERT_tokenizer)
    dataloader=DataLoader.DataLoader(dataset,batch_size=1,shuffle=False)
    data_itr=tqdm.tqdm(enumerate(dataloader),
                       total=len(dataloader),
                       bar_format="{l_bar}{r_bar}"
    )
    results=[]
    torch.manual_seed(0)
    for i,(text_id,input_id,label_id) in data_itr:
        text_id=text_id[0] # the dataloader output tuple
        input_id=input_id.squeeze()[1:-1] # the dataloader output 1*seq_len
        proposal=Prop_state(
            label_id,
            BERT_tokenizer,
            BERT,
            victim_classifier,
            # args.num_class,
            embed_search,
            )
        
        MH_sampler=MCMC(
            input_id,
            label_id,
            args.num_samples,
            args.Lambda,
            victim_classifier,
            rouge_sim,
            semantic_sim,
            BERT_tokenizer,
            BERT,
            proposal,
            args.num_class,
        )
        # print('the original input is:',input_id.shape)

        exmaple_ids, examples, rouge_value ,best_adv, best_index, best_rouge, success=MH_sampler.mcmc_iteration()
        results.append({'text_id':text_id,
                        'input_id':input_id,
                        'label_id':label_id,
                        'exmaple_ids':exmaple_ids,
                        'examples':examples,
                        'rouge_value':rouge_value,
                        'best_adv':best_adv,
                        'best_index':best_index,
                        'best_rouge':best_rouge,
                        'success':success
                        })
        torch.save(results,args.save_dir+args.dataset+'_checkpoints'+'.pt')
    torch.save(results,args.save_dir+args.dataset+'.pt')



def args_parser():
    parser=argparse.ArgumentParser()
    # parser.add_argument('--val', type=bool, default=False)
    parser.add_argument('--rouge_type', type=str, default='rouge1')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='results/')
    '''
    victim_model_name:
        1. ag_news
            --dataset: ag_news
            --victim_model_name: mrm8488/bert-mini-finetuned-age_news-classification
            --num_class: 4
            --Lambda: 0.19
            
        b. emotion
            --dataset: emotion
            --victim_model_name: bhadresh-savani/distilbert-base-uncased-emotion
            --num_class: 2
            --Lambda: 0.1
        b. SST-2
            --dataset: sst2
            --victim_model_name: philschmid/tiny-bert-sst2-distilled
            --num_class: 2
            --Lambda: 0.1
    '''
    parser.add_argument('--dataset', type=str, default='ag_news')
    parser.add_argument('--victim_model_name', type=str, default='mrm8488/bert-mini-finetuned-age_news-classification')
    parser.add_argument('--num_class', type=int, default=4)
    parser.add_argument('--Lambda', type=float, default=0.19)
    return parser.parse_args()

if __name__ == '__main__':
    args=args_parser()
    sampling_attacker(args)

# python main.py --dataset emotion --victim_model_name bhadresh-savani/distilbert-base-uncased-emotion --num_class 2 --Lambda 0.15
# python main.py --dataset sst2 --victim_model_name philschmid/tiny-bert-sst2-distilled --num_class 2 --Lambda 0.15
# nohup python -u main.py > ag_news.txt 2>&1 &
# nohup python -u main.py --dataset emotion --victim_model_name bhadresh-savani/distilbert-base-uncased-emotion --num_class 2 --Lambda 0.15 > emotion_iter.txt 2>&1 &
# nohup python -u main.py --dataset sst2 --victim_model_name philschmid/tiny-bert-sst2-distilled --num_class 2 --Lambda 0.15 > ss2_iter.txt 2>&1 &