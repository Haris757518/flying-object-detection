import json,time
from pathlib import Path
import requests
from websockets.sync.client import connect

base='http://127.0.0.1:8765';root=Path(__file__).resolve().parent
for _ in range(30):
    try:
        meta=requests.get(base+'/api/model',timeout=3).json()
        assert len(meta['techniques'])==6
        assert meta['technique_available']['lstm']
        assert meta['technique_available']['bilstm_attention']
        break
    except requests.ConnectionError:time.sleep(1)
report={}
for technique in ('yolo','cnn','transformer','cgan_transformer','lstm','bilstm_attention'):
    with (root/'demos/0.jpg').open('rb') as f:
        response=requests.post(base+'/api/detect',files={'file':('image.jpg',f,'image/jpeg')},data={'technique':technique},timeout=90)
    assert response.ok,response.text
    data=response.json();assert data['technique']==technique and data['detections']
    for d in data['detections']:
        assert d['detector_class'] and 0<=d['confidence']<=1
        assert bool(d['classifier'])==(technique!='yolo')
    job=requests.post(base+'/api/library/class-0-1',data={'technique':technique},timeout=15).json()['job_id']
    frames=[]
    with connect('ws://127.0.0.1:8765/api/live/'+job,max_size=5000000) as ws:
        while len(frames)<3:
            packet=json.loads(ws.recv(timeout=90))
            assert packet['type']=='frame',packet
            assert packet['technique']==technique
            frames.append(packet)
        requests.post(base+'/api/jobs/'+job+'/stop')
        while True:
            p=json.loads(ws.recv(timeout=90))
            if p['type']=='end':
                assert p['status']=='complete',p
                break
    report[technique]={'image_detections':data['detections'],'live_frames':len(frames),'live_tracks':frames[-1]['tracks']}
    print(technique,'PASS',[(d['class_name'],d['confidence']) for d in data['detections']],flush=True)
assert requests.post(base+'/api/library/class-0-1',data={'technique':'invalid'}).status_code==400
(root/'classifier-verification.json').write_text(json.dumps(report,indent=2))
