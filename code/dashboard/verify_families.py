"""Exercise both detector families and every trained classifier through HTTP/WS."""
import json
import os
import time
from pathlib import Path
import requests
from websockets.sync.client import connect

ROOT=Path(__file__).resolve().parent
BASE=os.environ.get('LAVEE_BASE_URL','http://127.0.0.1:8765').rstrip('/')
WS_BASE=BASE.replace('http://','ws://',1).replace('https://','wss://',1)
meta=requests.get(BASE+'/api/model',timeout=30).json()
assert all(f['available'] for f in meta['families'].values())
assert len(meta['families']['yolo12']['research']['fold_counts'])==10
report={}
for family in ('yolov8s','yolo12'):
    for method in ('yolo','cnn','transformer','cgan_transformer','lstm','bilstm_attention'):
        technique=('yolo12_' if family=='yolo12' else '')+method
        with (ROOT/'demos/0.jpg').open('rb') as f:
            response=requests.post(BASE+'/api/detect',files={'file':('sample.jpg',f,'image/jpeg')},data={'technique':technique,'confidence':.05},timeout=120)
        assert response.ok,response.text
        image=response.json()
        assert image['detections'],technique
        assert all(bool(r['classifier'])==(method!='yolo') for r in image['detections'])
        job=requests.post(BASE+'/api/library/class-0-1',data={'technique':technique,'confidence':.05},timeout=15).json()['job_id']
        with connect(WS_BASE+'/api/live/'+job,max_size=5000000) as ws:
            frames=[]
            while len(frames)<2:
                packet=json.loads(ws.recv(timeout=90))
                assert packet['type']=='frame',packet
                assert packet['technique']==technique
                frames.append(packet)
            requests.post(BASE+'/api/jobs/'+job+'/stop',timeout=15).raise_for_status()
            while True:
                p=json.loads(ws.recv(timeout=90))
                if p['type']=='end':
                    assert p['status']=='complete',p
                    break
        # Exercise live-camera inference with a supplied JPEG, without activating a physical camera.
        with connect(WS_BASE+'/api/webcam',max_size=5000000) as ws:
            ws.send(json.dumps({'technique':technique,'confidence':.05,'iou':.45}))
            ws.send((ROOT/'demos/0.jpg').read_bytes())
            cam=json.loads(ws.recv(timeout=90))
            assert cam['type']=='frame' and cam['tracks'],cam
        report[technique]={'image_detections':image['detections'],'live_frames':len(frames),'camera_frame':True}
        print(technique,'PASS',[(d['class_name'],d['confidence']) for d in image['detections']],flush=True)
assert requests.post(BASE+'/api/library/class-0-1',data={'technique':'yolo12_invalid'},timeout=15).status_code==400
(ROOT/'family-verification.json').write_text(json.dumps(report,indent=2))
