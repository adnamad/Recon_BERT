import json
from pathlib import Path
from typing import List
import numpy as np
import os
import PIL
import PIL.Image
import torchvision.transforms.functional as F
import torch.utils as utils
from torch.utils.data import Dataset
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize


def _convert_image_to_rgb(image):
    return image.convert("RGB")

class squarepad:
    def __init__(self,size):
        self.size=size
    def __call__(self,image):
        w,h=image.shape
        max_wh=max(w,h)
        hp=int((max_wh-w)/2)
        vp=int((max_wh-h)/2)
        padding=[hp,vp,hp,vp]
        return (F.pad(image,padding,0,"constant"))

class targetpad:
    def __init__(self,target_ratio,size):
        self.target_ratio=target_ratio
        self.size=size
    def __call__(self,image):
        w,h=image.size
        actual_ratio=max(w,h)/min(w,h)
        if  actual_ratio<self.target_ratio:
            return (image)
        scaled_max_wh=max(w,h)/self.target_ratio
        hp=max(int((scaled_max_wh-w)/2),0)
        vp=max(int((scaled_max_wh-h)/2),0)
        padding=[hp,vp,hp,vp]
        return (F.pad(image,padding,0,"constant"))

def squarepad_transform(dim):
    return (Compose([squarepad(dim),Resize(dim,interpolation=PIL.ImageBICUBIC),CenterCrop(dim),_convert_image_to_rgb,ToTensor(),
            Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))]))

def targetpad_transform(target_ratio,dim):
    return (Compose([targetpad(target_ratio,dim),Resize(dim,interpolation=PIL.ImageBICUBIC),CenterCrop(dim),_convert_image_to_rgb,ToTensor(),
            Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))]))



class fashion_iq(utils.data.Dataset):
    def __init__(self,root_path,split="train",dress_types=["dress","shirt","toptee"],mode="relative"):
        self.root_path=root_path
        self.mode=mode
        self.dress_types=dress_types
        self.split=split

        if mode not in ["relative","classic"]:
            raise ValueError("mode should be in relative or classic")
        if split not in ["train","test","val"]:
            raise ValueError("split should be in [test,train,val]")
        for dress_type in dress_types:

            if dress_type not in ["dress","shirt","val"]:
                raise ValueError("dress_type should be in ['dress','shirt','toptee']") 
        
        self.triplets=[]
        for dress_type in dress_types:
            path=os.path.join(self.root_path,"captions",f"cap.{dress_type}.{split}.json")
            with open(path) as f:
                self.triplets.extend(json.load(f))
        
        self.images_names=[]
        for dress_type in dress_types:
            path=os.path.join(self.root_path,"image_splits",f"split.{dress_type}.{split}.json")
            with open(path) as f:
                self.images_names.extend(json.load(f))
        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")


    def __len__(self):
        if self.mode=="relative":
            return (len(self.triplets))
        elif self.mode=="classic":
            return (len(self.images_names))
        else:
            raise ValueError("mode should be in ['relative','classic']")
    
    def __getitem__(self,idx):
        try:
            if self.mode=="relative":
                image_captions=self.triplets[idx]["captions"]
                reference_name=self.triplets[idx]["candidate"]
                if self.split=="train":
                    ref_img_path=os.path.join(self.root_path,"images",f"{reference_name}.png")
                    reference_image=PIL.Image.open(ref_img_path)
                    reference_image=ToTensor()(reference_image)

                    target_name=self.triplets[idx]["target"]
                    target_img_path=os.path.join(self.root_path,"images",f"{target_name}.png")
                    target_image=PIL.Image.open(target_img_path)
                    target_image=ToTensor()(target_image)

                    return(reference_image,target_image,image_captions)
                
                elif self.split=="val":
                    target_name=self.triplets[idx]["target"]
                    return (reference_name,target_name,image_captions)
                
                elif self.split=="test":
                    ref_img_path=os.path.join(self.root_path,"images",f"{reference_name}.png")
                    reference_image=PIL.Image.open(ref_img_path)
                    reference_image=ToTensor()(reference_image)
                    return (reference_name,reference_image,image_captions)
            
            elif self.mode=="classic":
                image_name=self.images_names[idx]
                image_path=os.path.join(self.root_path,"images",f"{image_name}.jpg")
                image=PIL.Image.open(image_path)
                image=ToTensor()(image)
                return (image_name,image)
        except Exception as e:
                print(f"Exception:{e}")
            
            
                


# root_path="/scratch/lxmert_data/"
# p=fashion_iq(root_path=root_path,split="train",dress_types=["shirt"],mode="relative")
# print(p.__len__())
# idx=np.random.randint(0,p.__len__())
# print(p[idx][0].shape)
# print(p[idx][1].shape)
# print(p[idx][2])



