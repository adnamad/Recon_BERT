import numpy as np
import json
import yaml
from pprint import pprint
import cv2
import torch
import torch.nn as nn
import torch.utils as utils 
import torch.nn.functional as F
import torch.optim as optim

#local imports
from models.lxmert_model import *
from data.dataloader import *
#import clip 


#seeds
seed=5
torch.manual_seed(5)
np.random.seed(5)


#model init
config_file="/home/srinjay/kitti360_reg/Recon_BERT/cfg/bert_config.json"
class BertConfig(object):
    def __init__(self,vocab_size_or_config_json_file,hidden_size=768,num_hidden_layers=12,num_attention_heads=12,intermediate_size=3072,hidden_act="gelu",hidden_dropout_prob=0.1,attention_probs_dropout_prob=0.1,max_position_embeddings=512,type_vocab_size=2,initializer_range=0.02):
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2 and isinstance(vocab_size_or_config_json_file, str)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

with open(config_file,"r",encoding="utf-8") as reader:
    text=reader.read()
model_config=BertConfig(vocab_size_or_config_json_file=-1)
for key,value in json.loads(text).items():
    model_config.__dict__[key]=value 

# lxmert_model=LXRTModel(model_config)
# print(lxmert_model)

lxmert_encoder=LXRTEncoder(model_config)
print(lxmert_encoder)


#dataloader
bs=16
data_root="/scratch/lxmert_data/"
traindataset=fashion_iq(root_path=data_root,split="train",dress_types=["shirt"],mode="relative")
train_dataloader=utils.data.DataLoader(traindataset,batch_size=8,shuffle=True,num_workers=4)

testdataset=fashion_iq(root_path=data_root,split="val",dress_types=["shirt"],mode="relative")
test_dataloader=utils.data.DataLoader(testdataset,batch_size=8,shuffle=False,num_workers=4)


#check fwd pass of lxmert-model.
idx=np.random.randint(0,len(traindataset))
image1=traindataset[idx][0]
image2=traindataset[idx][1]
text=traindataset[idx][2]
text1=text[0]
text2=text[1]
print(text)
print(text1)
print(text2)

#ques_id, feats, boxes, ques, target
#ques_id : integer
#feats : feature of patch #(batch_size,n_objects,feature_dim=2048)
#boxes : bbox coordinates of objs #(batch_size,n_objects,4)
#question : actual  list of question,string   
#label : answer to the question vqa

"""an example of coco"""
# {
# "answer_type": "other",
# "img_id": "COCO_train2014_000000458752",
# "label": {
#     "net": 1
# },
# "question_id": 458752000,
# "question_type": "what is this",
# "sent": "What is this photo taken looking through?"
# }

#generating a random id 
ques_id=np.random.randint(0,100000)
print(ques_id)