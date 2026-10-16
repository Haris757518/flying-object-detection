"""Rebuild AOD-4 + MBAOD without editing the original datasets.

Dependencies: python -m pip install numpy pillow
Run: python repair_merged_dataset.py --root "D:\\ai\\AI DETECTION MODEL"
"""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse, csv, hashlib, json, math, os, random, re, shutil
import numpy as np
from PIL import Image

ROOT=Path('D:/ai/AI DETECTION MODEL')
STAGE=ROOT/'.dataset_repair_work'
DEST=ROOT/'combined_dataset_fixed'
NAMES=['airplane','bird','helicopter','kite','parachute','UAV']
SOURCES=[('AOD4','AOD-4','train','train'),('AOD4','AOD-4','valid','valid'),('AOD4','AOD-4','test','test'),
         ('MBAOD','MBAOD','Train - V1 - Clean','Train - V1 - Clean'),('MBAOD','MBAOD','Vaild','Vaild'),('MBAOD','MBAOD','Test - Clean','Test - Clean')]
MAPS={'AOD4':{0:0,1:1,2:5,3:2},'MBAOD':dict(enumerate(range(6)))}
SPLITS=['train','val','test']; RATIOS=np.array([.70,.15,.15]); SEED=20261012

class DSU:
    def __init__(self,n): self.p=list(range(n)); self.size=[1]*n
    def find(self,x):
        while self.p[x]!=x: self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.size[a]<self.size[b]:a,b=b,a
        self.p[b]=a;self.size[a]+=self.size[b]

def iou(a,b):
    ax,ay,aw,ah=a[1:];bx,by,bw,bh=b[1:]
    overlap=max(0,min(ax+aw/2,bx+bw/2)-max(ax-aw/2,bx-bw/2))*max(0,min(ay+ah/2,by+bh/2)-max(ay-ah/2,by-bh/2))
    return overlap/(aw*ah+bw*bh-overlap+1e-12)

def discover():
    records=[]
    for tag,dataset,imgsplit,lblsplit in SOURCES:
        base=ROOT/'Flying_Object_Project/datasets'/dataset
        annotations='YOLOv8 format' if tag=='AOD4' else 'YOLOv8'
        image_dir=base/'Images'/imgsplit
        label_dir=base/'Annotations'/annotations/lblsplit/'labels'
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f'Required source folder missing: {image_dir} or {label_dir}')
        for p in sorted(image_dir.iterdir()):
            if p.suffix.lower() not in ['.jpg','.jpeg','.png']:continue
            label=label_dir/(p.stem+'.txt')
            records.append({'source':str(p),'source_label':str(label),'origin':tag,'original_split':imgsplit,'stem':tag+'_'+p.stem})
    assert records,'No sources'
    return records

DCT=np.cos(np.pi*(np.arange(32)[None,:]+.5)*np.arange(8)[:,None]/32)
def inspect(record):
    r=dict(record); boxes=[]; corrected=0
    try:
        p=Path(r['source_label'])
        if not p.is_file():raise ValueError('missing annotation')
        for line in p.read_text().splitlines():
            parts=line.split()
            if len(parts)!=5:raise ValueError('invalid label field count')
            old=int(parts[0]); vals=list(map(float,parts[1:]))
            if old not in MAPS[r['origin']] or not all(math.isfinite(x) for x in vals):raise ValueError('invalid class or nonfinite geometry')
            x,y,w,h=vals
            if w<=0 or h<=0:raise ValueError('nonpositive size')
            left,top,right,bottom=max(0,x-w/2),max(0,y-h/2),min(1,x+w/2),min(1,y+h/2)
            if right<=left or bottom<=top:raise ValueError('box outside image')
            new=[(left+right)/2,(top+bottom)/2,right-left,bottom-top]
            corrected+=int(max(abs(a-b) for a,b in zip(vals,new))>1e-8)
            boxes.append([MAPS[r['origin']][old],*new])
        data=Path(r['source']).read_bytes();r['sha256']=hashlib.sha256(data).hexdigest()
        from io import BytesIO
        with Image.open(BytesIO(data)) as im:
            im.load();r['dimensions']=list(im.size)
            small=im.convert('L').resize((32,32),Image.Resampling.LANCZOS)
            thumb=np.asarray(im.convert('L').resize((64,64),Image.Resampling.LANCZOS),dtype=np.uint8)
        low=DCT@np.asarray(small,dtype=np.float32)@DCT.T
        bits=low.flatten()>np.median(low.flatten()[1:]); bits[0]=False
        r['phash']=sum(int(v)<<i for i,v in enumerate(bits))
        r['boxes']=boxes;r['clipped_boxes']=corrected
        return r,thumb,None
    except Exception as e:return r,None,str(e)

def similarity(a,b):
    # Local SSIM on 8x8 tiles plus absolute luminance difference. Candidate verification only.
    a=a.astype(np.float32)/255;b=b.astype(np.float32)/255
    x=a.reshape(8,8,8,8).transpose(0,2,1,3).reshape(64,64)
    y=b.reshape(8,8,8,8).transpose(0,2,1,3).reshape(64,64)
    mx=x.mean(1);my=y.mean(1);vx=x.var(1);vy=y.var(1);cov=((x-mx[:,None])*(y-my[:,None])).mean(1)
    s=((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return float(s.mean()),float(np.abs(a-b).mean())

class BKTree:
    def __init__(self): self.nodes=[]
    def add(self,h,index):
        if not self.nodes:self.nodes.append([h,[index],{}]);return
        n=0
        while True:
            value,ids,children=self.nodes[n];d=(h^value).bit_count()
            if d==0:ids.append(index);return
            if d in children:n=children[d]
            else:children[d]=len(self.nodes);self.nodes.append([h,[index],{}]);return
    def query(self,h,radius):
        if not self.nodes:return []
        result=[];stack=[0]
        while stack:
            value,ids,children=self.nodes[stack.pop()];d=(h^value).bit_count()
            if d<=radius:result.extend(ids)
            stack.extend(node for edge,node in children.items() if d-radius<=edge<=d+radius)
        return result

def prepare():
    config={'root':str(ROOT),'output':str(DEST),'seed':SEED,'class_names':NAMES}
    if (STAGE/'manifest.json').exists():
        if json.loads((STAGE/'plan_config.json').read_text())!=config:
            raise RuntimeError('Existing work folder uses different settings; select another --work folder')
        print('Reusing completed preparation plan',flush=True);return
    records=discover(); print('Original clean images:',len(records),flush=True)
    STAGE.mkdir(parents=True,exist_ok=True)
    (STAGE/'plan_config.json').write_text(json.dumps(config,indent=2))
    source_signature=hashlib.sha256()
    for r in records:
        for key in ['source','source_label']:
            p=Path(r[key]);st=p.stat() if p.is_file() else None
            source_signature.update(f'{p}:{st.st_size if st else -1}:{st.st_mtime_ns if st else -1}'.encode())
    signature=source_signature.hexdigest()
    cache=STAGE/'inspection_cache.json';cache_array=STAGE/'inspection_thumbnails.npy'
    if cache.exists() and cache_array.exists() and json.loads(cache.read_text()).get('signature')==signature:
        saved=json.loads(cache.read_text());records=saved['records'];quarantine=saved['quarantine']
        thumbs=np.load(cache_array,mmap_mode='r');print('Resumed from completed image-check cache',flush=True)
    else:
        checked=[];thumbs=[];quarantine=[]
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i,(r,t,error) in enumerate(pool.map(inspect,records)):
                if error:quarantine.append({**r,'reason':error})
                else:checked.append(r);thumbs.append(t)
                if i and i%5000==0:print('Decoded and checked',i,'images',flush=True)
        records=checked
        np.save(cache_array,np.asarray(thumbs,dtype=np.uint8))
        cache.write_text(json.dumps({'signature':signature,'records':records,'quarantine':quarantine}))
        del checked
    dsu=DSU(len(records));source_keys={};exact={};removed=set();conflicts=[]
    for i,r in enumerate(records):
        original=Path(r['source']).stem.split('.rf.')[0]
        sequence=re.match(r'(V_[A-Z]+_\d+)_\d+',original) or re.match(r'(\d{8}_\d{6})_',original)
        key=r['origin']+':'+(sequence.group(1) if sequence else original)
        if key in source_keys:dsu.join(i,source_keys[key])
        else:source_keys[key]=i
        if r['sha256'] in exact:
            j=exact[r['sha256']];dsu.join(i,j)
            # Exact pixels: merge compatible annotations into one representative image.
            merged=list(records[j]['boxes']); conflict=False
            for box in r['boxes']:
                if any(int(box[0])!=int(other[0]) and iou(box,other)>.8 for other in merged):conflict=True
                if not any(int(box[0])==int(other[0]) and iou(box,other)>.8 for other in merged):merged.append(box)
            if conflict:conflicts.append([j,i]); removed.add(i);removed.add(j)
            else:records[j]['boxes']=merged;removed.add(i)
        else:exact[r['sha256']]=i
    print('Exact duplicate copies removed:',len(removed),'conflicting exact pairs:',len(conflicts),flush=True)
    tree=BKTree();similar_pairs=[];candidate_count=0
    for i,r in enumerate(records):
        if i in removed:continue
        for j in tree.query(r['phash'],3):
            if dsu.find(i)==dsu.find(j):continue
            candidate_count+=1
            score,mae=similarity(thumbs[i],thumbs[j])
            if score>=.98 and mae<=.025:
                dsu.join(i,j);similar_pairs.append([i,j,score,mae])
        tree.add(r['phash'],i)
        if i and i%5000==0:print('Similarity grouping',i,'verified pairs',len(similar_pairs),flush=True)
    del thumbs
    groups=defaultdict(list)
    for i,r in enumerate(records):
        if i not in removed:groups[dsu.find(i)].append(i)
    # Greedy grouped multi-class allocation, then improve by moving entire groups.
    vectors={g:np.array([len(ids),*np.bincount([int(b[0]) for i in ids for b in records[i]['boxes']],minlength=6)],dtype=float) for g,ids in groups.items()}
    total=sum(vectors.values()); target=RATIOS[:,None]*total[None,:]; denom=np.maximum(target,1)
    rng=random.Random(SEED);order=list(groups);rng.shuffle(order);order.sort(key=lambda g:vectors[g][0],reverse=True)
    counts=np.zeros_like(target);assign={}
    def loss(c):return float((((c-target)/denom)**2).sum())
    for g in order:
        v=vectors[g];s=min(range(3),key=lambda k:loss(counts+np.eye(3)[k,:,None]*v))
        assign[g]=s;counts[s]+=v
    for iteration in range(8):
        moves=0;rng.shuffle(order)
        for g in order:
            old=assign[g];v=vectors[g];best=old;bestloss=loss(counts)
            for s in range(3):
                if s==old:continue
                proposed=counts.copy();proposed[old]-=v;proposed[s]+=v
                score=loss(proposed)
                if score<bestloss-1e-10:best=s;bestloss=score
            if best!=old:counts[old]-=v;counts[best]+=v;assign[g]=best;moves+=1
        if not moves:break
    if any(counts[s,0]==0 or any(counts[s,1:]==0) for s in range(3)):raise RuntimeError('Grouping prevents usable class coverage in every split')
    manifest=[]
    for g,ids in groups.items():
        split=SPLITS[assign[g]];(STAGE/split/'labels').mkdir(parents=True,exist_ok=True)
        for i in ids:
            r=records[i];label=STAGE/split/'labels'/(r['stem']+'.txt')
            lines=[' '.join([str(int(b[0]))]+[format(float(x),'.12f') for x in b[1:]]) for b in r['boxes']]
            label.write_text('\n'.join(lines)+ ('\n' if lines else ''),encoding='utf-8')
            manifest.append({'source':r['source'],'source_label':r['source_label'],'original_split':r['original_split'],'origin':r['origin'],'image_name':r['stem']+Path(r['source']).suffix.lower(),'split':split,'group':str(g),'sha256':r['sha256'],'boxes':r['boxes'],'clipped_boxes':r['clipped_boxes']})
    if len({r['image_name'] for r in manifest})!=len(manifest):
        raise RuntimeError('Image filenames collide; review source naming before publication')
    with (STAGE/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['source','source_label','original_split','origin','image_name','split','group','sha256','clipped_boxes'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(manifest)
    with (STAGE/'similarity_pairs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['image_a','image_b','local_ssim','mean_absolute_difference'])
        for i,j,s,m in similar_pairs:w.writerow([records[i]['source'],records[j]['source'],s,m])
    (STAGE/'quarantine.json').write_text(json.dumps({'invalid_samples':quarantine,'conflicting_exact_pairs':[[records[a]['source'],records[b]['source']] for a,b in conflicts]},indent=2))
    summary={'seed':SEED,'class_names':NAMES,'source_images':len(records)+len(quarantine),'retained_images':len(manifest),'exact_duplicate_copies_or_conflicting_images_excluded':len(removed),'invalid_samples':len(quarantine),'conflicting_exact_pairs':len(conflicts),'clipped_boxes':sum(r['clipped_boxes'] for r in manifest),'negative_images':sum(not r['boxes'] for r in manifest),'groups':len(groups),'largest_group_images':max(map(len,groups.values())),'verified_similarity_pairs':len(similar_pairs),'similarity_candidates':candidate_count,'splits':{s:{'images':int(counts[k,0]),'boxes_by_class':dict(zip(NAMES,map(int,counts[k,1:])))} for k,s in enumerate(SPLITS)},'limitations':['MBAOD source-video identifiers are unavailable for numbered still images; filename grouping and conservative similarity checks cannot prove full scene independence.','Near-image grouping uses pHash distance <=3, 64px local SSIM >=0.98 and mean absolute difference <=0.025. All grouped non-exact images are retained.','Only canonical clean datasets are rebuilt. Weather-augmented train and adverse-weather test images are not pooled into this dataset.']}
    (STAGE/'audit.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)
    (STAGE/'data.yaml').write_text('path: "'+DEST.as_posix()+'"\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 6\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(NAMES)))
    (STAGE/'README.md').write_text('# Corrected combined airborne-object dataset\n\nBuilt from canonical AOD-4 and MBAOD clean image/YOLO annotation exports. The originals and old combined_dataset are unchanged.\n\nClass order: 0 airplane, 1 bird, 2 helicopter, 3 kite, 4 parachute, 5 UAV. AOD mapping: 0->0, 1->1, 2->5, 3->2.\n\nSource-video/timestamp groups, original filename variants, exact pixel duplicates and conservatively verified similar-image groups remain in one split. Target split proportions are 70/15/15, balanced by image and class-box counts, seed 20261012. Exact duplicates with compatible annotations are consolidated; all non-exact samples are retained. Empty labels are valid background samples. Box edges are clipped before normalization.\n\nSee audit.json, manifest.csv, similarity_pairs.csv, quarantine.json and verification.json for evidence. Missing MBAOD recording IDs prevent a guarantee of complete scene independence; investigate source-video metadata before claiming performance on entirely unseen recording sessions.\n\nDo not use these new test images for model selection. Models previously trained on the old dataset may have already seen them; train a fresh baseline and document pretraining provenance. Keep adverse-weather evaluation separate, and group clean/weather versions of the same image together when expanding the dataset.\n')
    # Completed-plan marker is written last, so an interrupted preparation is not reused.
    (STAGE/'manifest.json').write_text(json.dumps(manifest,indent=2))

def publish():
    manifest=json.loads((STAGE/'manifest.json').read_text())
    signature=hashlib.sha256((STAGE/'manifest.json').read_bytes()).hexdigest()
    marker=DEST/'.repair_in_progress.json'
    if DEST.exists():
        if (DEST/'verification.json').exists() and json.loads((DEST/'verification.json').read_text()).get('passed'):
            if not (DEST/'manifest.json').exists() or hashlib.sha256((DEST/'manifest.json').read_bytes()).hexdigest()!=signature:
                raise RuntimeError('Completed destination belongs to a different plan; choose a new --output')
            print('Dataset already completed; rerunning verification',flush=True);verify(DEST,manifest);return
        if not marker.exists() or json.loads(marker.read_text()).get('manifest_sha256')!=signature:
            raise RuntimeError('Destination exists and is not this unfinished repair; choose a new --output')
    needed=sum(Path(r['source']).stat().st_size for r in manifest)
    if shutil.disk_usage(ROOT).free<needed*1.05:raise RuntimeError('Insufficient destination disk space')
    DEST.mkdir(exist_ok=True)
    marker.write_text(json.dumps({'manifest_sha256':signature}))
    for split in SPLITS:
        (DEST/split/'images').mkdir(parents=True,exist_ok=True);shutil.copytree(STAGE/split/'labels',DEST/split/'labels',dirs_exist_ok=True)
    def copy(r):
        dst=DEST/r['split']/'images'/r['image_name']
        if not dst.exists() or hashlib.sha256(dst.read_bytes()).hexdigest()!=r['sha256']:
            shutil.copy2(r['source'],dst)
        if hashlib.sha256(dst.read_bytes()).hexdigest()!=r['sha256']:raise RuntimeError('Image copy hash mismatch: '+str(dst))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(copy,manifest)):
            if i and i%5000==0:print('Copied and hash-verified',i,'images',flush=True)
    for p in STAGE.iterdir():
        if p.is_file() and not p.name.startswith('inspection_'):shutil.copy2(p,DEST/p.name)
    # Publish a corrected regeneration script copy; preserve original notebook generator.
    original=ROOT/'generate_notebook.py'
    if original.is_file():
        patched=original.read_text().replace('aod_map = {0: 0, 1: 2, 2: 5, 3: 1}','aod_map = {0: 0, 1: 1, 2: 5, 3: 2}')
        patched=patched.replace('# AOD-4 classes: airplane, helicopter, drone, bird.', '# AOD-4 YOLO IDs: airplane, bird, drone, helicopter.')
        patched=patched.replace('# Here, assuming 0: airplane, 1: helicopter, 2: drone (UAV), 3: bird.', '# Verified source IDs: 0 airplane, 1 bird, 2 drone (UAV), 3 helicopter.')
        (DEST/'generate_notebook_mapping_corrected.py').write_text(patched,encoding='utf-8')
    if Path(__file__).resolve()!=DEST/'repair_merged_dataset.py':shutil.copy2(Path(__file__),DEST/'repair_merged_dataset.py')
    verify(DEST,manifest)
    marker.unlink()
    print('Published corrected dataset:',DEST,flush=True)

def verify(path,manifest):
    print(f'Starting final verification of {len(manifest):,} images and labels...',flush=True)
    seen={};groups=defaultdict(set);counts=Counter();boxes=Counter();errors=[]
    started=__import__('time').monotonic()
    def check_record(r):
        split=r['split'];p=path/split/'images'/r['image_name'];label=path/split/'labels'/(Path(r['image_name']).stem+'.txt')
        failures=[];classes=[];digest=None
        try:
            if not p.is_file() or not label.is_file():raise ValueError('missing image/label pair')
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            if digest!=r['sha256']:failures.append('image hash mismatch: '+str(p))
            actual=[]
            for line in label.read_text().splitlines():
                parts=line.split()
                if len(parts)!=5:raise ValueError('invalid label field count')
                c=int(parts[0]);x,y,w,h=map(float,parts[1:]);classes.append(c);actual.append([c,x,y,w,h])
                if c not in range(6) or not all(math.isfinite(t) for t in [x,y,w,h]) or w<=0 or h<=0 or min(x-w/2,y-h/2)<-1e-9 or max(x+w/2,y+h/2)>1+1e-9:
                    failures.append('invalid box: '+str(label))
            if len(actual)!=len(r['boxes']) or any(int(a[0])!=int(b[0]) or max(abs(x-y) for x,y in zip(a[1:],b[1:]))>1e-9 for a,b in zip(actual,r['boxes'])):
                failures.append('label differs from prepared manifest: '+str(label))
        except Exception as e:failures.append(f'{r["image_name"]}: {e}')
        return r,digest,classes,failures
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(r,digest,classes,failures) in enumerate(pool.map(check_record,manifest),1):
            split=r['split'];groups[r['group']].add(split);counts[split]+=1;errors.extend(failures)
            if digest is not None:
                if digest in seen:errors.append('duplicate image: '+r['image_name'])
                seen[digest]=split
            boxes.update((split,c) for c in classes)
            if i%2000==0 or i==len(manifest):
                elapsed=__import__('time').monotonic()-started
                print(f'Final verification: {i:,}/{len(manifest):,} ({i/len(manifest):.0%}), {elapsed:.0f}s, issues: {len(errors)}',flush=True)
    if any(len(s)>1 for s in groups.values()):errors.append('group crosses splits')
    for split in SPLITS:
        if len(list((path/split/'images').iterdir()))!=counts[split]:errors.append('unexpected image count')
        if len(list((path/split/'labels').iterdir()))!=counts[split]:errors.append('unexpected label count')
    result={'passed':not errors,'images':dict(counts),'total_images':sum(counts.values()),'unique_image_hashes':len(seen),'cross_split_group_leaks':sum(len(s)>1 for s in groups.values()),'invalid_or_missing_records':len(errors),'errors':errors[:30],'copy_hashes_verified':True}
    (path/'verification.json').write_text(json.dumps(result,indent=2))
    print('FINAL VERIFICATION',json.dumps(result),flush=True)
    if errors:raise RuntimeError('Dataset verification failed')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action',nargs='?',default='all',choices=['all','prepare','publish','verify'])
    ap.add_argument('--root',default=str(ROOT),help='AI DETECTION MODEL folder')
    ap.add_argument('--output',help='New destination; existing unrelated folders are never overwritten')
    ap.add_argument('--work',help='Work/cache folder for resumable preparation')
    args=ap.parse_args();ROOT=Path(args.root).resolve()
    DEST=Path(args.output).resolve() if args.output else ROOT/'combined_dataset_fixed'
    STAGE=Path(args.work).resolve() if args.work else ROOT/'.dataset_repair_work'
    if DEST==ROOT or DEST==ROOT/'combined_dataset' or ROOT in DEST.parents and 'Flying_Object_Project' in DEST.parts:
        raise SystemExit('Output must be a new folder, outside original datasets')
    if STAGE==DEST or STAGE==ROOT or STAGE==ROOT/'combined_dataset' or 'Flying_Object_Project' in STAGE.parts:
        raise SystemExit('Work folder must be separate from original datasets and destination')
    try:
        if args.action in ['all','prepare']:prepare()
        if args.action in ['all','publish']:publish()
        if args.action=='verify':
            manifest_path=DEST/'manifest.json'
            verify(DEST,json.loads(manifest_path.read_text()))
            marker=DEST/'.repair_in_progress.json'
            if marker.exists() and json.loads(marker.read_text()).get('manifest_sha256')==hashlib.sha256(manifest_path.read_bytes()).hexdigest():
                marker.unlink()
            print('Dataset verification completed:',DEST,flush=True)
    except KeyboardInterrupt:
        print('\nStopped. Originals are unchanged. Rerun the same command to reuse completed checkpoints.',flush=True)
        raise SystemExit(130)
