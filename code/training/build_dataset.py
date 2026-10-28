from pathlib import Path
import csv, json, random, hashlib, math, zipfile
from collections import Counter
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parent
SOURCE=Path('D:/ai/datasets/combined_dataset')
OUT=ROOT/'balanced_six_class'
NAMES=['airplane','bird','helicopter','kite','parachute','drone']
TARGETS={'train':500,'val':100,'test':100}
def main():
    OUT.mkdir(exist_ok=True)
    records=[]; hashes={}; group_splits={}
    with (SOURCE/'manifest.csv').open(newline='',encoding='utf-8-sig') as f:
        source_rows=list(csv.DictReader(f))
    for split,target in TARGETS.items():
        rows=[r for r in source_rows if r['split']==split]
        random.Random(42).shuffle(rows)
        counts=Counter()
        for c,name in enumerate(NAMES): (OUT/split/name).mkdir(parents=True,exist_ok=True)
        for row in rows:
            if all(counts[c]==target for c in range(6)):break
            image=SOURCE/split/'images'/row['image_name']
            label=SOURCE/split/'labels'/(image.stem+'.txt')
            boxes=[]
            for line in label.read_text().splitlines():
                t=line.split()
                if len(t)!=5:continue
                c=int(t[0]); vals=list(map(float,t[1:]))
                if c not in range(6) or counts[c]>=target:continue
                if not all(math.isfinite(v) for v in vals):continue
                boxes.append((c,*vals))
            if not boxes:continue
            # One object per source image increases diversity and avoids repeated crops.
            boxes.sort(key=lambda b:counts[b[0]])
            with Image.open(image) as opened:im=opened.convert('RGB')
            w,h=im.size
            for c,cx,cy,bw,bh in boxes:
                x1=max(0,int((cx-bw*.54)*w));y1=max(0,int((cy-bh*.54)*h))
                x2=min(w,math.ceil((cx+bw*.54)*w));y2=min(h,math.ceil((cy+bh*.54)*h))
                if x2-x1<12 or y2-y1<12:continue
                crop=ImageOps.pad(im.crop((x1,y1,x2,y2)),(224,224),color=(114,114,114))
                digest=hashlib.sha256(crop.tobytes()).hexdigest()
                if digest in hashes:continue
                group=str(row['group'])
                if group in group_splits and group_splits[group]!=split:raise RuntimeError('Group overlap')
                group_splits[group]=split;hashes[digest]=split
                filename=f'{split}/{NAMES[c]}/{digest[:24]}.jpg'
                crop.save(OUT/filename,quality=88)
                records.append({'file':filename,'class_id':c,'class_name':NAMES[c],'split':split,'group':group,'source_image':image.name,'source_sha256':row['sha256'],'crop_pixel_sha256':digest,'box_xyxy':json.dumps([x1,y1,x2,y2])})
                counts[c]+=1
                break
        assert all(counts[c]==target for c in range(6)),(split,counts)
        print(split,{NAMES[c]:counts[c] for c in range(6)},flush=True)
    with (OUT/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    groups_per_class={n:len({r['group'] for r in records if r['class_name']==n and r['split']=='train'}) for n in NAMES}
    summary={'classes':NAMES,'crops_per_class':TARGETS,'total_crops':len(records),'size':[224,224],'labels':'ground-truth YOLO boxes','train_groups_per_class':groups_per_class,'cross_split_group_overlap':0,'duplicate_crop_pixels':0,'limitations':['Some original recording IDs are unavailable; group independence is not a guarantee of recording-session independence.','Previously trained YOLO weights may have seen source images; crop classifier evaluation is not independent full-pipeline detector evaluation.','This balanced subset does not preserve the natural deployment class distribution.']}
    (OUT/'dataset.json').write_text(json.dumps(summary,indent=2))
    with zipfile.ZipFile(ROOT/'balanced_six_class.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
