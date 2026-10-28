"""Six-class spatial-patch recurrent classifiers; no temporal-motion claim."""
from pathlib import Path
import csv,json,argparse,random,hashlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from PIL import Image
from torchvision import transforms
import timm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from recurrent_models import RecurrentClassifier

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
TF=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])

class Crops(Dataset):
    def __init__(self,root,rows):self.root,self.rows=root,rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.root/self.rows[i]['file']) as im:x=TF(im.convert('RGB'))
        return x

def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)

def tokens(extractor,x):
    return nn.functional.adaptive_avg_pool2d(extractor.forward_features(x),(4,4)).flatten(2).transpose(1,2)

def fit(x,y,bi,epochs):
    seed();model=RecurrentClassifier(512,bidirectional=bi,attention=bi).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    loader=DataLoader(TensorDataset(x,y),batch_size=128,shuffle=True)
    for epoch in range(epochs):
        model.train();total=0
        for xb,yb in loader:
            loss=nn.functional.cross_entropy(model(xb.float().to(DEVICE)),yb.to(DEVICE))
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();total+=float(loss.detach())
        if epoch==0 or (epoch+1)%5==0:print(f'  epoch {epoch+1}/{epochs} loss {total/len(loader):.4f}',flush=True)
    return model

def evaluate(model,x,y):
    model.eval();pred=[]
    with torch.inference_mode():
        for batch in x.split(128):pred.extend(model(batch.float().to(DEVICE)).argmax(1).cpu().tolist())
    p,r,f,_=precision_recall_fscore_support(y.numpy(),pred,labels=list(range(6)),average='macro',zero_division=0)
    return {'accuracy':float(accuracy_score(y.numpy(),pred)),'macro_precision':float(p),'macro_recall':float(r),'macro_f1':float(f),'confusion_matrix':confusion_matrix(y.numpy(),pred,labels=list(range(6))).tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='/content/balanced_six_class');p.add_argument('--out',default='/content/recurrent_results');p.add_argument('--epochs',type=int,default=15);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if DEVICE!='cuda':raise RuntimeError('Enable a GPU runtime in Colab.')
    seed();root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((root/'manifest.csv').open()));classes=json.loads((root/'dataset.json').read_text())['classes']
    y=torch.tensor([int(r['class_id']) for r in rows]);groups=np.array([r['group'] for r in rows])
    indices={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ('train','val','test')}
    fingerprint=hashlib.sha256((root/'manifest.csv').read_bytes()).hexdigest()
    extractor=timm.create_model('resnet18',pretrained=True,num_classes=0).to(DEVICE).eval()
    for weight in extractor.parameters():weight.requires_grad_(False)
    cache=out.parent/'recurrent_feature_cache'/f'spatial_tokens_{fingerprint[:12]}.pt'
    cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():x=torch.load(cache,weights_only=True)
    else:
        features=[]
        with torch.inference_mode():
            for step,images in enumerate(DataLoader(Crops(root,rows),batch_size=64,num_workers=2)):
                with torch.autocast('cuda'):z=tokens(extractor,images.to(DEVICE))
                features.append(z.half().cpu())
                if step%10==0:print('Feature batches:',step+1,flush=True)
        x=torch.cat(features);torch.save(x,cache)
    assert x.shape==(len(rows),16,512),x.shape
    train=indices['train'];folds=list(StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=42).split(train,y[train].numpy(),groups[train]))
    report={'classes':classes,'pilot':a.pilot,'dataset_fingerprint':fingerprint,'scope':'Ground-truth crop classification using 16 spatial feature tokens. Not temporal motion or full-pipeline detector evaluation.','methods':{}}
    fold_files=[]
    for k,(tr,va) in enumerate(folds):
        assert not set(groups[train[tr]])&set(groups[train[va]])
        assert len(set(y[train[tr]].tolist()))==6 and len(set(y[train[va]].tolist()))==6
        fold_files.append({'fold':k+1,'training_files':[rows[i]['file'] for i in train[tr]],'validation_files':[rows[i]['file'] for i in train[va]]})
    (out/'folds.json').write_text(json.dumps(fold_files))
    for method,bi in [('LSTM',False),('BiLSTM_Attention',True)]:
        scores=[]
        def checkpoint(model,full=False):
            data={'head':{k:v.cpu() for k,v in model.state_dict().items()},'architecture':'resnet18','classes':classes,'input_basis':'spatial_tokens','grid_size':4,'input_dim':512,'hidden_dim':64,'bidirectional':bi,'attention':bi,'input_size':224,'normalization':'ImageNet mean/std','dataset_fingerprint':fingerprint}
            if full:data['extractor']={k:v.cpu() for k,v in extractor.state_dict().items()}
            return data
        for k,(tr,va) in enumerate(folds):
            if a.pilot and k>0:break
            print(method,'fold',k+1,flush=True)
            model=fit(x[train[tr]],y[train[tr]],bi,a.epochs)
            score=evaluate(model,x[train[va]],y[train[va]]);scores.append({'fold':k+1,**score})
            torch.save(checkpoint(model),out/f'{method}_fold{k+1}.pt')
            print('macro-F1',score['macro_f1'],flush=True)
        report['methods'][method]={'folds':scores,'mean_macro_f1':float(np.mean([s['macro_f1'] for s in scores])),'std_macro_f1':float(np.std([s['macro_f1'] for s in scores],ddof=1)) if len(scores)>1 else None}
        if not a.pilot:
            print(method,'final fit',flush=True);model=fit(x[train],y[train],bi,a.epochs)
            report['methods'][method].update(validation=evaluate(model,x[indices['val']],y[indices['val']]),test=evaluate(model,x[indices['test']],y[indices['test']]))
            torch.save(checkpoint(model,True),out/f'{method}_final.pt')
        (out/'metrics.json').write_text(json.dumps(report,indent=2))
    print('Saved',out,flush=True)
if __name__=='__main__':main()
