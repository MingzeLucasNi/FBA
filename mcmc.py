import torch
from torch.nn import functional as F
import copy




class Prop_state:
    '''
    Usage:
        proposing a state
    Input:
        input_id: the input id of the input sentence
        label_id: the label id of the input sentence
        BERT_tokenizer: the tokenizer of BERT
        BERT: the BERT model
    Output:
        prop_state: the proposed state
    '''
    def __init__(self, label, tokenizer, model, victim_model,embed_prob):
        # self.input_id = input_id
        self.label = label
        self.tokenizer = tokenizer
        self.model = model
        self.victim_model = victim_model
        self.embed_prob = embed_prob

        self.insert=0.1
        self.sub=0.8
        self.remove=0.1
        self.remove_hold=0.2
        

        # self.num_class=num_class

        self.mask_id=50264
        self.start_id=0
        self.end_id=2
        self.emd_weight=0.8
        self.bert_weight=0.05
        self.bert_top_weight=0.15
    
    def action_sample(self, input_ids): 
        '''
        Usage:
            sample the actions(insert,substitution,remove) for the given input sequence x.
        Input: 
            None
        Output:
            action: the action to be taken. Notice: 0 for insertion, 1 for substitution, 2 for removal.
            prob: the probability of the action.
        '''
        if input_ids.shape[0]<3:
            dis=torch.tensor([self.insert+self.remove,self.sub])
            action=torch.multinomial(dis,1)
            prob=dis[action]
        else:
            dis=torch.tensor([self.insert,self.sub,self.remove])
            action=torch.multinomial(dis,1)
            prob=dis[action]
        return action,prob
    
    def position_sample(self,input_ids):
        '''
        Usage: 
            sample the position of the token to take action. NOTICE: if the action is insertion, the insertion position is the one infron of sampled position, therefore we wont insert token at the end of a sentence.
        Input: 
            input_ids: the input sentence(squeezed 1*token_num) without special tokens.[cls] [sep]
        Output:
            position: the position of the token to be masked
            prob: [FLOAT], the probability of sampling this position.
        '''
        pos_dis=self.f_logit_dis(input_ids)
        # pos_dis=torch.full((input_ids.shape[0],),1/input_ids.shape[0])
        position=torch.multinomial(pos_dis,1)
        prob=pos_dis[position]
        return position, prob, pos_dis

    def BERT_dis(self,input_ids,position):
        start=torch.tensor([self.start_id])
        end=torch.tensor([self.end_id])
        new_id=torch.cat((start,input_ids,end))
        logits=self.model(new_id.unsqueeze(0), labels=new_id.unsqueeze(0))['logits'].squeeze()[position+1,:]
        dis=F.softmax(logits,dim=0).squeeze()
        return logits, dis
    
    def hybrid_dis(self, input_ids, position, attacked_id):
        _,bert_dis=self.BERT_dis(input_ids,position)
        
        sort_bert_in  =torch.sort(bert_dis,descending=True).indices.squeeze()[:4]
        if attacked_id not in sort_bert_in:
            sort_bert_in[-1]=attacked_id
        top_bert_dis=torch.zeros(len(self.tokenizer))
        for i in sort_bert_in:
            top_bert_dis[i]=0.2


        embed_ind=self.embed_prob.prob(attacked_id)
        embed_dis=torch.zeros(len(self.tokenizer))
        for i in embed_ind:
            embed_dis[i]=0.2
        hybrid_dis=self.bert_weight*bert_dis+self.bert_top_weight*top_bert_dis+self.emd_weight*embed_dis
        return hybrid_dis



    
    def f_logit_dis(self,input_ids):
        position_weights=[]
        ori_text=self.tokenizer.decode(input_ids,skip_special_tokens=True)
        original_logit=self.victim_model.logits(ori_text,self.label)
        for i in range(input_ids.shape[0]):
            can = torch.cat((input_ids[:i], input_ids[i+1:])).squeeze()
            text=self.tokenizer.decode(can,skip_special_tokens=True)
            logit=self.victim_model.logits(text,self.label)
            logit_drop=original_logit-logit
            position_weights.append(logit_drop)
        position_weights=torch.tensor(position_weights)
        pos_dis=F.softmax(position_weights,dim=0).squeeze()
        return pos_dis
        

            
    def remove_token(self,input_ids):
        '''
        Usage:
            for removing the token at the given position.
        Input:
            input_ids: the input sentence(squeezed 1*token_num)
            position: the position of the token to be removed.
        Output:
            new_state_id: the new sentence after removing the token.
        '''
        u=torch.randn(1)
        if u<self.remove_hold:
            position,position_prob, _ =self.position_sample(input_ids)
            limit=input_ids.shape[0]-1
            c=0
            while position==limit:
                position,position_prob, alter =self.position_sample(input_ids)
                c+=1
                if c>100:
                    position=limit-1
                    position_prob=alter[position]
                    break
            new_state=torch.cat((input_ids[:position], input_ids[position+1:]))
            proposing_prob=self.remove*self.remove_hold*position_prob


            #calculate the reverse probability
            attacked_id=input_ids[position]
            a, b ,reverse_pos_dis=self.position_sample(new_state)
            reverse_pos_prob=reverse_pos_dis[position]
            reverse_mask_state=input_ids
            reverse_mask_state[position]=self.mask_id
            _,reverse_dis=self.BERT_dis(reverse_mask_state,position)
            token_id=torch.multinomial(reverse_dis,1)
            reverse_p=reverse_dis[token_id]
            reverse_prob=self.insert*reverse_pos_prob*reverse_p
        
        else:
            new_state=input_ids
            proposing_prob=self.remove*(1-self.remove_hold)
            reverse_prob=proposing_prob
            new_state=input_ids
        return new_state,proposing_prob,reverse_prob
    
    def insert_token(self,input_ids):
        position,position_prob,_=self.position_sample(input_ids)
        mask_sentence=torch.cat((input_ids[:position],torch.tensor([self.mask_id]),input_ids[position:]))
        logits,dis=self.BERT_dis(mask_sentence,position)
        token_id=torch.multinomial(dis,1)
        token_prob=dis[token_id]
        new_state=torch.cat((input_ids[:position],token_id,input_ids[position:]))
        proposing_prob=self.insert*position_prob*token_prob

        #calculate the reverse probability
        reverse_position,reverse_pos_prob,_=self.position_sample(new_state)
        reverse_prob=self.remove*reverse_pos_prob*self.remove_hold

        return new_state,proposing_prob,reverse_prob

    def substitution_token(self,input_ids):
        position,position_prob,_=self.position_sample(input_ids)
        attacked_id=input_ids[position]
        mask_sentence=torch.cat((input_ids[:position],torch.tensor([self.mask_id]),input_ids[position+1:]))
        dis=self.hybrid_dis(mask_sentence,position,attacked_id)
        token_id=torch.multinomial(dis,1)
        token_prob=dis[token_id]
        new_state=torch.cat((input_ids[:position],token_id,input_ids[position+1:]))
        proposing_prob=self.sub*position_prob*token_prob

        #calculate the reverse probability
        reverse_dis=self.hybrid_dis(mask_sentence,position,token_id)
        reverse_token_prob=reverse_dis[attacked_id]
        reverse_prob=self.sub*position_prob*reverse_token_prob

        return new_state,proposing_prob,reverse_prob


    def propose_state(self,old_state):
        '''
        Usage:
            Propose a new state( {e,l,o} in the paper), and calculate the probability of the proposed state. q(x(t+1)|x(t)) in the paper.
        Input:
            old_state: the old state of the sentence.
        Output:
            new_state: the new state of the sentence.
            prob: the probability of the proposed state.
        '''
        action,action_prob=self.action_sample(old_state)
        if action==0:
            new_state,proposing_prob,reverse_prob=self.insert_token(old_state)


        elif action==1:
            new_state,proposing_prob,reverse_prob=self.substitution_token(old_state)

        elif action==2:
            new_state,proposing_prob,reverse_prob=self.remove_token(old_state)
        return new_state,proposing_prob,reverse_prob


    






class MCMC:
    '''
    Usage: 
        generate the $Num_sample$ adversarial samples for the given input sequence x.
        --------------------------------------------------------------------------------------------------------------------
    Input: 
        input_ids:[TENSOR](length of inputs sequence) the id for input sequence, with length of input_id.(with special tokens [cls],[sep])
        label: [INTEGER] the corret label for the input sequence.
        Num_sample: [INTEGER] the number of samples or iteration will mcmc runs
        Lambda: [FLOAT] the lambda to calculate the pi(x) (adversarial distribution)
        victim_model: [CLASS] the victim model to be attacked and we have only the logits of the model.
        rouge: [CLASS]the rouge to calculate the ALIGNMENT similarity between the proposed sentence and the original sentence.
        sentence_similarity:[CLASS] the semantic similarity between the proposed sentence and the original sentence.
        BERT_tokenizer: [CLASS] the tokenizer for BERT. This tokenizer is based on HuggingFace's BERT tokenizer.
        BERT:[CLASS] the BERT model to provide the insert and sub token. This model is based on HuggingFace's BERT model.
        --------------------------------------------------------------------------------------------------------------------

    Output: 
        exmaple_ids: [LIST], the list of the adversarial samples dictionary ID.
        examples: [LIST], the list of the adversarial samples.
        pi_value: [LIST], the list of the adversarial samples pi(x) value.
        best_adv: [STRING], the best adversarial sample from samples.(the best is defined with the largest adversarial density,pi(x))
        --------------------------------------------------------------------------------------------------------------------
    '''
    def __init__(self,input_ids, label, Num_samples, Lambda, victim_model, rouge, sentence_similarity,BERT_tokenizer, BERT, Prop_state, num_class):
        self.input_ids    =input_ids # remove the special tokens [cls] [sep]
        self.label        =label
        self.Num_samples  = Num_samples
        self.Lambda       = Lambda
        self.victim_model = victim_model
        self.rouge        = rouge
        self.sem_similarity = sentence_similarity
        self.BERT_tokenizer = BERT_tokenizer
        self.BERT = BERT
        self.propose = Prop_state
        self.num_class = num_class
    def adversarial_dis_prob(self,text_id):
        '''
        Usage:
            Calculate the probability of the adversarial distribution, pi(x).
        Input:
            text_id: the input sentence(squeezed 1*token_num) without special tokens.[cls] [sep]
        Output:
            prob: the probability of the adversarial distribution.
        '''
        ori_text=self.BERT_tokenizer.decode(self.input_ids)
        pro_text=self.BERT_tokenizer.decode(text_id)

        # pi=self.Lambda*(1-self.victim_model.prob(pro_text,self.label))+(1-self.Lambda)/2*(self.sem_similarity.sem(pro_text,ori_text)+self.rouge.Rouge(pro_text,ori_text))

        # prop_drop=1-self.victim_model.prob(pro_text,self.label)
        # adjust=prop_drop*(prop_drop>1/self.num_class)+(1/self.num_class-0.05)*(prop_drop<=1/self.num_class)
        # pi=self.Lambda*adjust+(1-self.Lambda)*self.sem_similarity.sem(pro_text,ori_text)
        
        pi=self.Lambda*(1-self.victim_model.prob(pro_text,self.label))+(1-self.Lambda)*self.sem_similarity.sem(pro_text,ori_text)
        
        return pi

        
    def accept_rate(self, old_state, new_state,proposing_prob, reverse_prob):
        '''
        Usage:
            Calculate the acceptance rate of the proposed state.
        Input:
            old_state: the old state of the sentence.
            new_state: the new state of the sentence.
            proposing_prob: the probability of the proposed state.
            reverse_prob: the probability of reversing the proposal to the original state.
        Output:
            accept_rate: the acceptance rate of the proposed state.
        '''
        old_pi=self.adversarial_dis_prob(old_state)
        new_pi=self.adversarial_dis_prob(new_state)
        accept_rate=min(1,reverse_prob*new_pi/old_pi/proposing_prob)
        return accept_rate


    
    def mcmc_iteration(self):
        '''
        Usage:
            MCMC iteration for sampling Num_sample from the adversarial distribution.
        Input:
            The input from this class.
        Output:
            The output from this class.
        '''
        old_state=copy.copy(self.input_ids)
        exmaple_ids=[]
        examples=[]
        pi_value=[]
        rouge_value=[]
        success=[]
        best_rouge=False
        best_index=False
        best_adv=False
        print('Start MCMC sampling...')
        # print('Input ids',self.input_ids)
        for i in range(self.Num_samples):
            new_state,proposing_prob,reverse_prob=self.propose.propose_state(old_state)
            accept_rate=self.accept_rate(old_state, new_state,proposing_prob, reverse_prob)

            if accept_rate>torch.rand(1):
                old_state=new_state
            else:
                old_state=old_state
            decoded_text=self.BERT_tokenizer.decode(old_state)
            ad_success=self.victim_model.predict(decoded_text)!=self.label
            rouge_score=self.rouge.Rouge(decoded_text,self.BERT_tokenizer.decode(self.input_ids))

            success.append(ad_success)
            rouge_value.append(rouge_score)
            exmaple_ids.append(old_state)
            examples.append(decoded_text)
            # pi_value.append(self.adversarial_dis_prob(old_state))
            if ad_success and rouge_score>best_rouge:
                best_adv=decoded_text
                best_rouge=rouge_score
                best_index=i

            if (i+1)%15==0:
                print('=========================================')

                print('##### The number of iteration ###### :',i)
                # print('##### The new state ###### :',old_state.shape)
                print('##### The adversarial text ###### :',decoded_text)
                print('##### The rouge score ###### :',rouge_score)
                print('##### The best index ###### :', best_index)
                print('##### The best adversarial example ###### :', best_adv)
                print('##### The successful attack ######:',ad_success)
                print('##### The best rouge ###### :',best_rouge)
                print('number of successful attack:', sum(success))
                print('=========================================')
            if (i+1)% 30==0 and (1 in success):
                break
        # print(torch.tensor(pi_value).shape)
        return exmaple_ids, examples, rouge_value ,best_adv, best_index, best_rouge, success
