from pathlib import Path
import threading
import math
import json
import cv2
import torch
from torch import nn
from PIL import Image, ImageOps
from torchvision import transforms
import timm
import yolo12_backend
from recurrent_models import RecurrentClassifier

ROOT=Path(__file__).resolve().parent/'research/classification_results'
FILES={'cnn':'CNN_final.pt','transformer':'Transformer_final.pt','cgan_transformer':'cGAN_Transformer_final.pt','lstm':'LSTM_final.pt','bilstm_attention':'BiLSTM_Attention_final.pt'}
LABELS={'yolo':'YOLO only','cnn':'YOLO + CNN','transformer':'YOLO + Transformer','cgan_transformer':'YOLO + cGAN-trained Transformer','lstm':'YOLO + LSTM (spatial patches)','bilstm_attention':'YOLO + BiLSTM + Attention (spatial patches)'}
TRANSFORM=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
CACHE={}
LOCK=threading.Lock()

def validate(technique):
    if technique.startswith('yolo12_'):
        if not yolo12_backend.ready():raise ValueError('YOLO12 package has not passed validation.')
        if technique[7:] not in LABELS:raise ValueError('Unknown classification technique.')
        return
    if technique not in LABELS:raise ValueError('Unknown classification technique.')
    if technique!='yolo' and not (ROOT/FILES[technique]).is_file():raise ValueError('Six-class trained weights are not installed for this technique.')

def load(technique):
    if technique not in CACHE:
        checkpoint=torch.load(ROOT/FILES[technique],map_location='cpu',weights_only=True)
        extractor=timm.create_model(checkpoint['architecture'],pretrained=False,num_classes=0)
        extractor.load_state_dict(checkpoint['extractor'],strict=True)
        if technique in ('lstm','bilstm_attention'):
            if checkpoint.get('input_basis')!='spatial_tokens' or checkpoint.get('grid_size')!=4 or checkpoint.get('input_dim')!=512:
                raise ValueError('Incompatible recurrent checkpoint: expected 16 spatial tokens with 512 features.')
            expected=technique=='bilstm_attention'
            if checkpoint['bidirectional']!=expected or checkpoint['attention']!=expected:raise ValueError('Recurrent architecture mismatch.')
            head=RecurrentClassifier(512,bidirectional=expected,attention=expected)
        else:
            head=nn.Linear(checkpoint['head']['weight'].shape[1],len(checkpoint['classes']))
        if checkpoint['classes']!=['airplane','bird','helicopter','kite','parachute','drone']:raise ValueError('Checkpoint class order does not match dashboard.')
        head.load_state_dict(checkpoint['head'],strict=True)
        CACHE[technique]=(extractor.eval(),head.eval(),checkpoint['classes'])
    return CACHE[technique]

def classify(image,boxes,technique):
    validate(technique)
    if technique.startswith('yolo12_'):
        key=technique[7:]
        return [None]*len(boxes) if key=='yolo' or not boxes else yolo12_backend.classify(image,boxes,key)
    if technique=='yolo' or not boxes:return [None]*len(boxes)
    h,w=image.shape[:2];crops=[]
    for box in boxes:
        x1,y1,x2,y2=box;dx=(x2-x1)*.04;dy=(y2-y1)*.04
        left=max(0,int(x1-dx));top=max(0,int(y1-dy));right=min(w,math.ceil(x2+dx));bottom=min(h,math.ceil(y2+dy))
        if right<=left or bottom<=top:raise ValueError('Invalid detector crop.')
        crop=Image.fromarray(cv2.cvtColor(image[top:bottom,left:right],cv2.COLOR_BGR2RGB))
        crops.append(TRANSFORM(ImageOps.pad(crop,(224,224),color=(114,114,114))))
    with LOCK,torch.inference_mode():
        extractor,head,names=load(technique)
        batch=torch.stack(crops)
        if technique in ('lstm','bilstm_attention'):
            z=nn.functional.adaptive_avg_pool2d(extractor.forward_features(batch),(4,4)).flatten(2).transpose(1,2)
        else:z=extractor(batch)
        probs=head(z).softmax(1)
    return [{'class_id':int(row.argmax()),'class_name':names[int(row.argmax())],'confidence':round(float(row.max()),4)} for row in probs]

def annotate(result,records):
    image=result.plot(labels=False)
    for record in records:
        x,y=map(int,record['box'][:2]);prefix=f"ID {record['track_id']} · " if record.get('track_id') is not None else ''
        label=f"{prefix}{record['class_name']} {record['confidence']:.2f}"
        if record.get('classifier'):label+=f" / YOLO {record['detector_class']}"
        label=label.replace('·','|')
        width=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.5,1)[0][0]
        cv2.rectangle(image,(max(0,x),max(0,y-24)),(min(image.shape[1],x+width+8),max(24,y)),(35,65,40),-1)
        cv2.putText(image,label,(max(0,x+3),max(17,y-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    return image

def research_summary():
    report=json.loads((ROOT/'metrics.json').read_text())
    return {'scope':report['scope'],'classes':report['classes'],'methods':{name:{'folds':len(data['folds']),'cv_macro_f1':data['mean_macro_f1'],'cv_std':data['std_macro_f1'],'test_accuracy':data['test']['accuracy'],'test_macro_f1':data['test']['macro_f1']} for name,data in report['methods'].items()}}
