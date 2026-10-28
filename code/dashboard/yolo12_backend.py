"""Runtime matching the deadline Colab notebook's Fold 1 export."""
from pathlib import Path
import csv
import json
import math
import statistics
import threading
import cv2
import torch
import timm
from torch import nn
from PIL import Image
from torchvision import transforms
from yolo12_heads import make_head

ROOT = Path(__file__).resolve().parent / 'models/yolo12_fold1'
FILES = {'cnn':'CNN_head.pt', 'lstm':'LSTM_head.pt', 'transformer':'Transformer_head.pt',
         'bilstm_attention':'BiLSTM_Attention_head.pt', 'cgan_transformer':'cGAN_Transformer_head.pt'}
METHODS = {'cnn':'CNN', 'lstm':'LSTM', 'transformer':'Transformer',
           'bilstm_attention':'BiLSTM_Attention', 'cgan_transformer':'cGAN_Transformer'}
REQUIRED = ['yolo12_best.pt', 'resnet18_frozen.pt', 'run_config.json', 'cv_progress.json', 'cv_summary.csv', *FILES.values()]
TF = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                         transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
CACHE = {}
LOCK = threading.Lock()

def ready():
    return (ROOT/'validated.json').is_file() and all((ROOT/f).is_file() and (ROOT/f).stat().st_size for f in REQUIRED)

def config():
    return json.loads((ROOT/'run_config.json').read_text())

def names():
    return ['drone' if n.lower()=='uav' else n.lower() for n in config()['names']]

def load(technique):
    if 'extractor' not in CACHE:
        ex = timm.create_model('resnet18', pretrained=False, num_classes=0)
        ex.load_state_dict(torch.load(ROOT/'resnet18_frozen.pt', map_location='cpu', weights_only=True), strict=True)
        CACHE['extractor'] = ex.eval()
    if technique not in CACHE:
        head = make_head(METHODS[technique])
        head.load_state_dict(torch.load(ROOT/FILES[technique], map_location='cpu', weights_only=True), strict=True)
        CACHE[technique] = head.eval()
    return CACHE['extractor'], CACHE[technique], names()

def classify(image, boxes, technique):
    if not boxes:return []
    h,w = image.shape[:2]
    crops = []
    for x1,y1,x2,y2 in boxes:
        # Same tight boxes and resize as notebook inference; no old 8% context/padding.
        l,t,r,b = max(0,int(round(x1))),max(0,int(round(y1))),min(w,int(round(x2))),min(h,int(round(y2)))
        if r<=l or b<=t:raise ValueError('Invalid detector crop.')
        crops.append(TF(Image.fromarray(cv2.cvtColor(image[t:b,l:r],cv2.COLOR_BGR2RGB))))
    with LOCK, torch.inference_mode():
        ex,head,classes = load(technique)
        tokens = nn.functional.adaptive_avg_pool2d(ex.forward_features(torch.stack(crops)),(4,4)).flatten(2).transpose(1,2)
        probs = head(tokens).softmax(1)
    return [{'class_id':int(p.argmax()),'class_name':classes[int(p.argmax())], 'confidence':round(float(p.max()),4)} for p in probs]

def research_summary():
    folds = json.loads((ROOT/'cv_progress.json').read_text())
    rows = list(csv.DictReader((ROOT/'cv_summary.csv').open()))
    detector = {}
    for metric in ('mAP50','mAP50_95','precision','recall'):
        values = [f['yolo_detector'][metric] for f in folds]
        detector[metric] = {'mean':statistics.mean(values),'std':statistics.stdev(values),'values':values}
    methods = {}
    for name in METHODS.values():
        recorded = next(r for r in rows if r['model']==name)
        scores = [f['crop_classifiers'][name] for f in folds]
        methods[name] = {'folds':len(folds), 'cv_macro_f1':float(recorded['mean']), 'cv_std':float(recorded['sample_std']),
                         'fold_metrics':scores, 'fold_values':[s['macro_f1'] for s in scores]}
    return {'scope':'Fold 1 demo weights. Ten-fold detector and ground-truth crop results; no ten-fold end-to-end validation or final locked-test result.',
            'group_basis':config()['group_basis'], 'classes':names(), 'detector':detector, 'methods':methods,
            'fold_counts':[{'fold':f['fold'],'train_images':f['n_train_images'],'validation_images':f['n_validation_images']} for f in folds]}
