import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, SmoothL1Loss

BertLayerNorm=torch.nn.LayerNorm
def swish(x):
    return (x*torch.sigmoid(x))
ACT2FN = {"gelu":F.gelu,"relu":F.relu,"swish":swish}


class bert_embedding(torch.nn.Module):
    def __init__(self,cfg):
        super(bert_embedding,self).__init__()   
        self.word_embeddings=torch.nn.Embedding(cfg.vocab_size,cfg.hidden_size,padding_idx=0)
        self.position_embeddings=torch.nn.Embedding(cfg.max_position_embeddings,cfg.hidden_size,padding_idx=0)
        self.token_type_embeddings=torch.nn.Embedding(cfg.type_vocab_size,cfg.hidden_size,padding_idx=0)
        self.LayerNorm=BertLayerNorm(cfg.hidden_size,eps=1e-12)
        self.dropout=torch.nn.Dropout(cfg.hidden_dropout_prob)
    
    def forward(self,input_ids,token_type_ids=None):
        seq_length=input_ids.size(1)
        position_ids=torch.arange(seq_length,dtype=torch.long,device=input_ids.device)
        position_ids=position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids=torch.zeros_like(input_ids)
        word_embeddings=self.word_embeddings(input_ids)
        position_embedding=self.position_embeddings(position_ids)
        token_type_embeddings=self.token_type_embeddings(token_type_ids)
        embeddings=word_embeddings+position_embedding+token_type_embeddings
        embeddings=self.LayerNorm(embeddings)
        embeddings=self.dropout(embeddings)
        return (embeddings)
        

class bert_attention(torch.nn.Module):
    def __init__(self,cfg,ctx_dim=None):
        super().__init__()
        if cfg.hidden_size%cfg.num_attention_heads!=0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention""heads (%d)"%(cfg.hidden_size,cfg.num_attention_heads))
        self.num_attnetion_heads=cfg.num_attention_heads
        self.attention_head_size=int(cfg.hidden_size/cfg.num_attention_heads)
        self.all_head_size=self.num_attnetion_heads*self.attention_head_size

        if ctx_dim is None:
            ctx_dim=cfg.hidden_size
        self.query=torch.nn.Linear(cfg.hidden_size,self.all_head_size)
        self.key=torch.nn.Linear(ctx_dim,self.all_head_size)
        self.value=torch.nn.Linear(ctx_dim,self.all_head_size)
        self.dropout=torch.nn.Dropout(cfg.attention_probs_dropout_prob)
    
    def transpose_for_scores(self,x):
        new_x_shape=x.size()[:-1] + (self.num_attnetion_heads,self.attention_head_size)
        x=x.view(*new_x_shape)
        return (x.permute(0,2,1,3))
    
    def forward(self,hidden_states,context,attention_mask=None):
        mixed_query_layer=self.query(hidden_states)
        mixed_key_layer=self.key(context)
        mixed_value_layer=self.value(context)

        query_layer=self.transpose_for_scores(mixed_query_layer)
        key_layer=self.transpose_for_scores(mixed_key_layer)
        value_layer=self.transpose_for_scores(mixed_value_layer)

        attention_scores=torch.matmul(query_layer,key_layer.transpose(-1,-2))
        attention_scores=attention_scores/math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores=attention_scores+attention_mask
        attention_probs=torch.nn.softmax(dim=-1)(attention_scores)
        context_layer=torch.matmul(attention_probs,value_layer)
        context_layer=context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape=context_layer.size()[:-2] + (self.all_head_size,)#bs,n_patches,n_heads*heads_dim
        context_layer=context_layer.view(*new_context_layer_shape)
        return (context_layer)

class bert_attnoutput(torch.nn.Module):
    def __init__(self,cfg):
        super(bert_attnoutput,self).__init__()
        self.dense=torch.nn.Linear(cfg.hidden_size,cfg.hidden_size)
        self.LayerNorm=BertLayerNorm(cfg.hidden_size,eps=1e-12)
        self.dropout=torch.nn.Dropout(cfg.hidden_dropout_prob)
    
    def forward(self,hidden_states,input_tensor):
        hidden_states=self.dense(hidden_states)
        hidden_states=self.dropout(hidden_states)
        hidden_states=self.LayerNorm(hidden_states+input_tensor)
        return (hidden_states)

class bert_crossattnlayer(torch.nn.Module):
    def __init__(self,cfg):
        super(bert_crossattnlayer,self).__init__()
        self.att=bert_attention(cfg)
        self.output=bert_attnoutput(cfg)
    
    def forward(self,input_tensor,ctx_tensor,ctx_att_mask=None):
        output=self.att(input_tensor,ctx_tensor,ctx_att_mask)
        attention_output=self.output(output,input_tensor)
        return (attention_output)


class bert_selfattnlayer(torch.nn.Module):
    def __init__(self,cfg):
        super(bert_selfattnlayer,self).__init__()
        self.att=bert_attention(cfg)
        self.output=bert_attnoutput(cfg)
    
    def forward(self,input_tensor,att_mask=None):
        output=self.att(input_tensor,input_tensor,att_mask)
        attention_output=self.output(output,input_tensor)
        return (attention_output)

class bert_intermediate(torch.nn.Module):
    def __init__(self,cfg):
        super(bert_intermediate,self).__init__()
        self.dense=torch.nn.Linear(cfg.hidden_size,cfg.intermediate_size)
        if isinstance(cfg.hidden_act,str) or (sys.version_info[0]==2 and isinstance(cfg.hidden_act,unicode)):
            self.intermediate_act_fn=ACT2FN[cfg.hidden_act]
        else:
            self.intermediate_act_fn=cfg.hidden_act
    
    def forward(self,hidden_states):
        hidden_states=self.dense(hidden_states)
        hidden_states=self.intermediate_act_fn(hidden_states)
        return (hidden_states)

class bert_output(torch.nn.Module):
    def __init__(self,cfg):
        super(bert_output,self).__init__()
        self.dense=torch.nn.Linear(cfg.intermediate_size,cfg.hidden_size)
        self.LayerNorm=BertLayerNorm(cfg.hidden_size,eps=1e-12)
        self.dropout=torch.nn.Dropout(cfg.hidden_dropout_prob)
    
    def forward(self,hidden_states,input_tensor):
        hidden_states=self.dense(hidden_states)
        hidden_states=self.dropout(hidden_states)
        hidden_states=self.LayerNorm(hidden_states+input_tensor)
        return (hidden_states)

class bertlayer(torch.nn.Module):
    def __init__(self,cfg):
        super(bertlayer,self).__init__()
        self.attention=bert_selfattnlayer(cfg)
        self.intermediate=bert_intermediate(cfg)
        self.output=bert_output(cfg)
    
    def forward(self,hidden_states,attention_mask):
        attention_output=self.attention(hidden_states,attention_mask)
        intermediate_output=self.intermediate(attention_output)
        layer_output=self.output(intermediate_output,attention_output)
        return (layer_output)


class LXRTXLayer(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        #cross in upper block
        self.visual_attention=bert_crossattnlayer(cfg)
        #self attention for both modalities
        self.lang_self_att=bert_selfattnlayer(cfg)
        self.visn_self_att=bert_selfattnlayer(cfg)
        #inter and output for lang and vis
        self.lang_inter=bert_intermediate(cfg)
        self.lang_output=bert_output(cfg)
        self.visn_inter=bert_intermediate(cfg)
        self.visn_output=bert_output(cfg)
    
    def cross_att(self,lang_input,lang_attention_mask,visn_input,visn_attention_mask):
        lang_att_output=self.visual_attention(lang_input,visn_input,ctx_att_mask=visn_attention_mask)#input is lang, context is visn
        vis_att_output=self.visual_attention(visn_input,lang_input,ctx_att_mask=lang_attention_mask)#input is visn, context is lang
        return (lang_att_output,vis_att_output)
    
    def self_att(self,lang_input,lang_attention_mask,visn_input,visn_attention_mask):
        lang_att_output=self.lang_self_att(lang_input,lang_attention_mask)
        visn_att_output=self.visn_self_att(visn_input,visn_attention_mask)
        return (lang_att_output,visn_att_output)
    
    def output_fc(self,lang_input,visn_input):
        #fc
        lang_inter_output=self.lang_inter(lang_input)
        visn_inter_output=self.visn_inter(visn_input)

        #layer output
        lang_output=self.lang_output(lang_inter_output,lang_input)
        visn_output=self.lang_output(visn_inter_output,visn_input)
        return (lang_output,visn_output)
    
    def forward(self,lang_feats,lang_attn_mask,visn_feats,visn_attn_mask):
        lang_att_output=lang_feats
        visn_att_output=visn_feats
        
        lang_att_output,visn_att_output=self.cross_att(lang_att_output,lang_attn_mask,visn_att_output,visn_attn_mask)
        lang_att_output,visn_att_output=self.self_att(lang_att_output,lang_attn_mask,visn_att_output,visn_attn_mask)
        lang_output,visn_output=self.output_fc(lang_att_output,visn_att_output)
        return (lang_output,visn_output)
    

#this is a placeholder for object features. in the experiment the patch features and patch pos embeddings  will come from the CLIP encoder.
class VisualFeatEncoder(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        feat_dim=cfg.visual_feat_dim
        pos_dim=cfg.visual_pos_dim
        
        #object encoding
        self.visn_fc=torch.nn.Linear(feat_dim,cfg.hidden_size)
        self.visn_layer_norm=BertLayerNorm(cfg.hidden_size,eps=1e-12)
        #box encoding
        self.box_fc=torch.nn.Linear(pos_dim,cfg.hidden_size)
        self.box_layer_norm=BertLayerNorm(cfg.hidden_size,eps=1e-12)
        self.dropout=torch.nn.Dropout(cfg.hidden_dropout_prob)
    
    def forward(self,vis_input):
        feats,boxes=vis_input
        x=self.visn_fc(feats)
        x=self.visn_layer_norm(x)
        y=self.box_fc(boxes)
        y=self.box_layer_norm(y)
        return((x+y)/2)


class LXRTEncoder(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        #encode objects
        self.visn_fc=VisualFeatEncoder(cfg)
        #bert layers
        #the number is borrowed from lxmert paper.
        self.num_l_layers=cfg.l_layers#check config
        self.num_x_layers=cfg.x_layers#check config
        self.num_r_layers=cfg.r_layers#check config

        self.layer=torch.nn.ModuleList([bertlayer(cfg) for _ in range(self.num_l_layers)])#language
        self.x_layers=torch.nn.ModuleList([LXRTXLayer(cfg) for _ in range(self.num_x_layers)])#cross
        self.r_layers=torch.nn.ModuleList([bertlayer(cfg) for _ in range(self.num_r_layers)])#image/relation
    
    def forward(self,lang_feats,lang_attention_mask,visn_feats,visn_attention_mask=None):
        #first get image embeddings and then language embeddings but bert extracts language embeddings before this module separately.
        visn_feats=self.visn_fc(visn_feats)

        #language layers
        for layer in self.layer:
            lang_feats=layer(lang_feats,lang_attention_mask)
        #image/relation layers
        for layer in self.r_layers:
            visn_feats=layer(visn_feats,visn_attention_mask)
        #cross-modality layers
        for layer in self.x_layers:
            lang_feats,visn_feats=layer(lang_feats,lang_attention_mask,visn_feats,visn_attention_mask)
        return(lang_feats,visn_feats)

class bertpooler(torch.nn.Module):
    def __init__(self,cfg):
        super(bertpooler,self).__init__()
        self.dense=torch.nn.Linear(cfg.hidden_size,cfg.hidden_size)
        self.activation=torch.nn.Tanh()
    
    def forward(self,hidden_states):
        first_token_tensor=hidden_states[:,0]
        pooled_output=self.dense(first_token_tensor)
        pooled_output=self.activation(pooled_output)
        return (pooled_output)


class LXRTModel(torch.nn.Module):
    def __init__(self,cfg):
        super(LXRTModel,self).__init__()
        self.cfg=cfg
        self.embeddings=bert_embedding(cfg)#get lang embeddings,for image embeddings are obtained within the LXRTEncoder 
        self.encoder=LXRTEncoder(cfg)
        self.pooler=bertpooler(cfg)
        self.apply(self.init_bert_weights)
    
    def init_bert_weights(self,module):
        if isinstance(module,(torch.nn.Linear,torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0,std=self.cfg.initializer_range)
        elif isinstance(module,BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module,torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self,input_ids,token_type_ids=None,attention_mask=None,visual_feats=None,visual_attention_mask=None):
        if attention_mask is None:
            attention_mask=torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids=torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask=extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask=(1.0-extended_attention_mask)*-10000.0

        if visual_attention_mask is not None:
            extended_visual_attention_mask=visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask=extended_visual_attention_mask.to(dtype=next(self.parameters()).dtype)#fp16 compatibility
            extended_visual_attention_mask=(1.0-extended_visual_attention_mask)*-10000.0
        else:
            extended_visual_attention_mask=None
        
        #lang embedding
        embedding_output=self.embeddings(input_ids,token_type_ids)

        #lxrt backbone this will also have the image embedding
        lang_feats,visn_feats=self.encoder(embedding_output,extended_attention_mask,visn_feats=visual_feats,visn_attention_mask=extended_visual_attention_mask)

        #pool
        pooled_output=self.pooler(lang_feats)
        return((lang_feats,visn_feats),pooled_output)
    

class lxrt_featextraction(torch.nn.Module):
    def __init__(self,cfg,mode="lxr"):
        super(lxrt_featextraction,self).__init__()
        self.bert=LXRTModel(cfg)
        self.mode=mode
        self.apply(self.init_bert_weights)
        
    def init_bert_weights(self,module):
        if isinstance(module,(torch.nn.Linear,torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0,std=self.cfg.initializer_range)
        elif isinstance(module,BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module,torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self,input_ids,token_type_ids,attention_mask=None,visual_feats=None,visual_attention_mask=None):
        feat_seq,pooled_output=self.bert(input_ids,token_type_ids,attention_mask,visual_feats=visual_feats,visual_attention_mask=visual_attention_mask)

        if self.mode=="x":
            return (pooled_output) #yellow block in model figure (see paper)
        elif "x" in self.mode and ("l" in self.mode or "r" in self.mode):
            return (feat_seq,pooled_output)
        elif "l" in self.mode or "r" in self.mode:
            return (feat_seq)
        else:
            return (None)
