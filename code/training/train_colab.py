"""Frozen CNN/ViT features, grouped 10-fold heads, and conditional-GAN augmentation."""
from pathlib import Path
import argparse, csv, json, random, hashlib
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
NORMALIZE=transforms.Normalize((.485,.456,.406),(.229,.224,.225))
IMAGE_TRANSFORM=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),NORMALIZE])

def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)

class Crops(Dataset):
    def __init__(self,root,rows,transform=IMAGE_TRANSFORM):self.root,self.rows,self.transform=root,rows,transform
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(self.root/r['file']) as im:x=self.transform(im.convert('RGB'))
        return x,int(r['class_id'])

def features(model,loader):
    output=[];model.eval()
    with torch.inference_mode():
        for x,_ in loader:
            with torch.autocast(device_type=DEVICE,enabled=DEVICE=='cuda'):
                z=model(x.to(DEVICE))
            output.append(z.float().cpu())
    return torch.cat(output)

def fit_head(x,y,epochs):
    seed(42)
    head=nn.Linear(x.shape[1],6).to(DEVICE)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.01)
    loader=DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=256,shuffle=True)
    for _ in range(epochs):
        head.train()
        for xb,yb in loader:
            loss=nn.functional.cross_entropy(head(xb.to(DEVICE)),yb.to(DEVICE))
            optimizer.zero_grad();loss.backward();optimizer.step()
    return head

def evaluate(head,x,y):
    with torch.inference_mode():pred=head(x.to(DEVICE)).argmax(1).cpu().numpy()
    p,r,f,_=precision_recall_fscore_support(y.numpy(),pred,labels=list(range(6)),average='macro',zero_division=0)
    return {'accuracy':float(accuracy_score(y.numpy(),pred)),'macro_precision':float(p),'macro_recall':float(r),'macro_f1':float(f),'confusion_matrix':confusion_matrix(y.numpy(),pred,labels=list(range(6))).tolist()}

class Generator(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(6,16)
        def block(a,b):return [nn.ConvTranspose2d(a,b,4,2,1,bias=False),nn.BatchNorm2d(b),nn.ReLU(True)]
        self.net=nn.Sequential(nn.ConvTranspose2d(116,256,4,1,0,bias=False),nn.BatchNorm2d(256),nn.ReLU(True),*block(256,128),*block(128,64),*block(64,32),nn.ConvTranspose2d(32,3,4,2,1),nn.Tanh())
    def forward(self,z,y):return self.net(torch.cat([z,self.embedding(y)],1)[:,:,None,None])

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(6,64*64)
        self.net=nn.Sequential(nn.Conv2d(4,32,4,2,1),nn.LeakyReLU(.2),nn.Conv2d(32,64,4,2,1),nn.LeakyReLU(.2),nn.Conv2d(64,128,4,2,1),nn.LeakyReLU(.2),nn.Conv2d(128,256,4,2,1),nn.LeakyReLU(.2),nn.Conv2d(256,1,4))
    def forward(self,x,y):return self.net(torch.cat([x,self.embedding(y).view(-1,1,64,64)],1)).flatten()

def train_gan(root,rows,epochs,out):
    seed(42);g=Generator().to(DEVICE);d=Discriminator().to(DEVICE)
    tf=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((.5,)*3,(.5,)*3)])
    loader=DataLoader(Crops(root,rows,tf),batch_size=64,shuffle=True,num_workers=2)
    go=torch.optim.Adam(g.parameters(),lr=.0002,betas=(.5,.999));do=torch.optim.Adam(d.parameters(),lr=.0002,betas=(.5,.999))
    criterion=nn.BCEWithLogitsLoss();history=[]
    for epoch in range(epochs):
        for real,y in loader:
            real,y=real.to(DEVICE),y.to(DEVICE);b=len(y)
            fake=g(torch.randn(b,100,device=DEVICE),y)
            do.zero_grad();dl=criterion(d(real,y),torch.ones(b,device=DEVICE)*.9)+criterion(d(fake.detach(),y),torch.zeros(b,device=DEVICE));dl.backward();do.step()
            go.zero_grad();gl=criterion(d(fake,y),torch.ones(b,device=DEVICE));gl.backward();go.step()
        history.append({'epoch':epoch+1,'generator_loss':float(gl.detach()),'discriminator_loss':float(dl.detach())})
        print(f'cGAN {epoch+1}/{epochs}: G={history[-1]["generator_loss"]:.3f} D={history[-1]["discriminator_loss"]:.3f}',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    torch.save(g.cpu().state_dict(),out/'generator.pt');g.to(DEVICE).eval()
    (out/'losses.json').write_text(json.dumps(history,indent=2))
    with torch.inference_mode():
        labels=torch.arange(6,device=DEVICE).repeat_interleave(4)
        samples=(g(torch.randn(24,100,device=DEVICE),labels)+1)/2
        from torchvision.utils import save_image
        save_image(samples,out/'generated_samples.jpg',nrow=4)
    return g

def synthetic_features(g,extractor,per_class):
    ys=torch.arange(6).repeat_interleave(per_class);zs=[]
    with torch.inference_mode():
        for labels in ys.split(64):
            x=(g(torch.randn(len(labels),100,device=DEVICE),labels.to(DEVICE))+1)/2
            x=nn.functional.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
            x=NORMALIZE(x)
            zs.append(extractor(x).float().cpu())
    return torch.cat(zs),ys

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='/content/balanced_six_class');parser.add_argument('--out',default='/content/classification_results');parser.add_argument('--folds',type=int,default=10);parser.add_argument('--epochs',type=int,default=30);parser.add_argument('--gan-epochs',type=int,default=10);parser.add_argument('--synthetic-per-class',type=int,default=100);parser.add_argument('--cgan',action='store_true');parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    if DEVICE!='cuda':print('WARNING: No GPU detected. Enable a Colab GPU runtime.',flush=True)
    seed(42);root=Path(args.data);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((root/'manifest.csv').open()))
    classes=json.loads((root/'dataset.json').read_text())['classes']
    train_idx=np.array([i for i,r in enumerate(rows) if r['split']=='train']);val_idx=np.array([i for i,r in enumerate(rows) if r['split']=='val']);test_idx=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    y=torch.tensor([int(r['class_id']) for r in rows]);groups=np.array([r['group'] for r in rows])
    folds=list(StratifiedGroupKFold(n_splits=args.folds,shuffle=True,random_state=42).split(train_idx,y[train_idx].numpy(),groups[train_idx]))
    for tr,va in folds:
        assert not set(groups[train_idx[tr]]) & set(groups[train_idx[va]])
        assert len(set(y[train_idx[tr]].tolist()))==6
        assert len(set(y[train_idx[va]].tolist()))==6,'A fold lacks a class; revise grouping or subset.'
    fold_manifest=[{'fold':k+1,'training_files':[rows[i]['file'] for i in train_idx[tr]],'validation_files':[rows[i]['file'] for i in train_idx[va]]} for k,(tr,va) in enumerate(folds)]
    (out/'folds.json').write_text(json.dumps(fold_manifest))
    fingerprint=hashlib.sha256((root/'manifest.csv').read_bytes()).hexdigest()
    report={'classes':classes,'pilot':args.pilot,'dataset_fingerprint':fingerprint,'methods':{},'scope':'Classification of ground-truth crops. Not full detector/pipeline cross-validation.'}
    for method,architecture in [('CNN','resnet18'),('Transformer','deit_tiny_patch16_224')]:
        print('Loading pretrained',architecture,flush=True)
        extractor=timm.create_model(architecture,pretrained=True,num_classes=0).to(DEVICE).eval()
        for p in extractor.parameters():p.requires_grad_(False)
        cache=out/f'{architecture}_{fingerprint[:12]}_features.pt'
        if cache.exists():x=torch.load(cache,weights_only=True)
        else:
            loader=DataLoader(Crops(root,rows),batch_size=64,num_workers=2,shuffle=False)
            x=features(extractor,loader);torch.save(x,cache)
        runs=[];gan_runs=[]
        for k,(tr,va) in enumerate(folds):
            if args.pilot and k>0:break
            ti,vi=train_idx[tr],train_idx[va]
            head=fit_head(x[ti],y[ti],args.epochs);score=evaluate(head,x[vi],y[vi]);runs.append({'fold':k+1,**score});print(method,'fold',k+1,score['macro_f1'],flush=True)
            torch.save({'head':head.cpu().state_dict(),'architecture':architecture,'classes':classes},out/f'{method}_fold{k+1}.pt')
            if method=='Transformer' and args.cgan:
                g=train_gan(root,[rows[i] for i in ti],args.gan_epochs,out/f'cgan_fold{k+1}')
                sx,sy=synthetic_features(g,extractor,args.synthetic_per_class)
                head=fit_head(torch.cat([x[ti],sx]),torch.cat([y[ti],sy]),args.epochs)
                score=evaluate(head,x[vi],y[vi]);gan_runs.append({'fold':k+1,**score});print('cGAN + Transformer',k+1,score['macro_f1'],flush=True)
                torch.save({'head':head.cpu().state_dict(),'architecture':architecture,'classes':classes},out/f'cGAN_Transformer_fold{k+1}.pt')
                del g
        report['methods'][method]={'folds':runs}
        if gan_runs:report['methods']['cGAN + Transformer']={'folds':gan_runs}
        if not args.pilot:
            head=fit_head(x[train_idx],y[train_idx],args.epochs)
            report['methods'][method].update(validation=evaluate(head,x[val_idx],y[val_idx]),test=evaluate(head,x[test_idx],y[test_idx]))
            torch.save({'head':head.cpu().state_dict(),'extractor':extractor.cpu().state_dict(),'architecture':architecture,'classes':classes,'input_size':224,'normalization':'ImageNet mean/std','dataset_fingerprint':fingerprint},out/f'{method}_final.pt');extractor.to(DEVICE)
            if method=='Transformer' and args.cgan:
                g=train_gan(root,[rows[i] for i in train_idx],args.gan_epochs,out/'cgan_final');sx,sy=synthetic_features(g,extractor,args.synthetic_per_class)
                head=fit_head(torch.cat([x[train_idx],sx]),torch.cat([y[train_idx],sy]),args.epochs)
                report['methods']['cGAN + Transformer'].update(validation=evaluate(head,x[val_idx],y[val_idx]),test=evaluate(head,x[test_idx],y[test_idx]))
                torch.save({'head':head.cpu().state_dict(),'extractor':extractor.cpu().state_dict(),'architecture':architecture,'classes':classes,'input_size':224,'normalization':'ImageNet mean/std','dataset_fingerprint':fingerprint},out/'cGAN_Transformer_final.pt')
        for entry in report['methods'].values():
            scores=entry['folds']
            entry['mean_macro_f1']=float(np.mean([r['macro_f1'] for r in scores]));entry['std_macro_f1']=float(np.std([r['macro_f1'] for r in scores],ddof=1)) if len(scores)>1 else None
        (out/'metrics.json').write_text(json.dumps(report,indent=2))
        del extractor
        if torch.cuda.is_available():torch.cuda.empty_cache()
    print('Saved results:',out,flush=True)
if __name__=='__main__':main()
