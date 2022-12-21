import os
import torch
import torch.nn as nn
from tokenizer import BertTokenizer
from  lxmert_model import lxrt_featextraction

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,input_ids,input_mask,segment_ids):
        self.input_ids=input_ids
        self.input_mask=input_mask
        self.segment_ids=segment_ids

def convert_sents_to_features(sents,max_seq_length,tokenizer):
    features=[]
    for (i,sent) in enumerate(sents):
        tokens_a=tokenizer.tokenize(sent.strip())
    
    # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a)>max_seq_length-2:
            tokens_a=tokens_a[:(max_seq_length-2)]  

        # Keep segment id which allows loading BERT-weights.
        tokens=["[CLS]"]+tokens_a+["[SEP]"]
        segment_ids=[0]*len(tokens)
        input_ids=tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask=[1]*len(input_ids)
        # Zero-pad up to the sequence length.
        padding=[0]*(max_seq_length-len(input_ids))
        input_ids+=padding
        input_mask+=padding
        segment_ids+=padding

        assert len(input_ids)==max_seq_length
        assert len(input_mask)==max_seq_length
        assert len(segment_ids)==max_seq_length

        features.append(InputFeatures(input_ids=input_ids,input_mask=input_mask,segment_ids=segment_ids))
    return (features)



class lxmertencoder(torch.nn.Module):
    def __init__(self,cfg,max_seq_length,mode="x"):
        super().__init__()
        self.cfg=cfg
        self.max_seq_length=max_seq_length
        self.tokenizer=BertTokenizer.from_pretrained("bert_base_uncased",do_lower_case=True)
        self.model=lxrt_featextraction(cfg,mode=mode)#did not load bert pretrained weights
        self.model.apply(self.model.init_bert_weights)
    
    @property
    def dim(self):
        return (self.cfg.hidden_size)
    
    def save(self,path):
        torch.save(self.model_state_dict(),os.path.join("%s_LXRT.pth"%path))
    
    def load(self,path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s"%path)
        state_dict=torch.load("%s_LXRT.pth"%path)
        new_state_dict={}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]]=value
            else:
                new_state_dict[key]=value
        state_dict=new_state_dict
        # Print out the differences of pre-trained and model weights.
        load_keys=set(state_dict.keys())
        model_keys=set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self,sents,feats,visual_attention_mask=None):
        train_features=convert_sents_to_features(sents,self.max_seq_length,self.tokenizer)
        input_ids=torch.tensor([f.input_ids for f in train_features],dtype=torch.long).cuda()
        input_mask=torch.tensor([f.input_mask for f in train_features],dtype=torch.long).cuda()
        segment_ids=torch.tensor([f.segment_ids for f in train_features],dtype=torch.long).cuda()

        output=self.model(input_ids,segment_ids,input_mask,visual_feats=feats,visual_attention_mask=visual_attention_mask)
        return (output)





