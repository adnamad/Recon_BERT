# Recon_BERT

## Project Doc
Project details with experiments [here](https://docs.google.com/document/d/1aQyJ4R7a1U1tSIOxoBDfknMo7oPmgC5kw3E17XylDhI/edit?usp=sharing)

## Dataset 
Fashion IQ is a dataset to facilitate research on natural language based interactive image retrieval.
The images can be downloaded from [here](https://github.com/hongwang600/fashion-iq-metadata). 

The image attribute features can be downloaded from [here](https://ibm.box.com/s/imyukakmnrkk2zuitju2m8akln3ayoct).


## Model Architecture 
[OpenAI CLIP](https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py)

[LXMERT](https://github.com/airsplay/lxmert/blob/0db1182b9030da3ce41f17717cc628e1cd0a95d5/src/lxrt/modeling.py)


## To-Do

- [ ] Extract intermediate CLIP resent features (Adnan)
- [ ] LXMERT integration with FIQ 
- [ ] Image Masking logic 
- [ ] Implement Decoder
- [ ] Loss and final integration
- [ ] Train and Eval
