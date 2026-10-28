import json
import time
from pathlib import Path
import cv2
import requests
from websockets.sync.client import connect

root=Path(__file__).resolve().parent
manifest=json.loads((root/'demos/manifest.json').read_text())
for item in manifest:
    if item['class_id']>=0:
        assert len(item['clips'])>=2
    for clip in item['clips']:
        cap=cv2.VideoCapture(str(root/'demos'/clip['file']))
        ok,frame=cap.read()
        assert ok,clip
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT)>20,clip
        cap.release()
base='http://127.0.0.1:8765'
for _ in range(30):
    try:
        assert requests.get(base+'/api/model',timeout=2).ok
        break
    except requests.ConnectionError:
        time.sleep(1)
j=requests.post(base+'/api/library/mixed-1',timeout=20).json()['job_id']
classes=set();packets=[]
with connect('ws://127.0.0.1:8765/api/live/'+j,max_size=5000000) as ws:
    while len(packets)<12:
        p=json.loads(ws.recv(timeout=40))
        assert p['type']=='frame',p
        classes.update(t['class_name'] for t in p['tracks'])
        packets.append({'frame':p['frame'],'classes':[t['class_name'] for t in p['tracks']]})
    requests.post(base+'/api/jobs/'+j+'/stop')
    while True:
        p=json.loads(ws.recv(timeout=40))
        if p['type']=='end':
            assert p['status']=='complete',p
            break
assert len(classes)>=2,classes
report={'clips_verified':sum(len(x['clips']) for x in manifest),'mixed_classes_detected':sorted(classes),'live_packets':packets}
(root/'library-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
