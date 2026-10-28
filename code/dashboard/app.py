from pathlib import Path
import base64
import json
import threading
import time
import uuid
import subprocess
import imageio_ffmpeg
import sys
import asyncio
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import classifiers
import yolo12_backend
import torch
torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
def load_model(family='yolov8s'):
    detector = YOLO(str(yolo12_backend.ROOT/'yolo12_best.pt' if family=='yolo12' else ROOT/'models/best.pt'))
    detector.model.names = {key: 'drone' if name.lower() == 'uav' else name
                            for key, name in detector.names.items()}
    if family=='yolo12' and list(detector.names.values()) != yolo12_backend.names():
        raise ValueError('YOLO and classifier class order mismatch.')
    return detector

model = load_model()
models = {'yolov8s':model}

def selected_model(technique):
    family = 'yolo12' if technique.startswith('yolo12_') else 'yolov8s'
    if family not in models:models[family]=load_model(family)
    return models[family]

def inference_size(technique):
    return 416 if technique.startswith('yolo12_') else 640
lock = threading.Lock()
app = FastAPI(title='LAVEE · Flying Object Detection')
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
app.mount('/demos', StaticFiles(directory=ROOT / 'demos'), name='demos')
(ROOT / 'results').mkdir(exist_ok=True)
app.mount('/results', StaticFiles(directory=ROOT / 'results'), name='results')
jobs = {}
pool = ThreadPoolExecutor(max_workers=1)

@app.get('/')
def home():
    return FileResponse(ROOT / 'static/index.html')

@app.get('/api/model')
def metadata():
    manifest = ROOT / 'demos/manifest.json'
    active = yolo12_backend.ready()
    return {'model': 'YOLOv8s', 'classes': model.names,
            'families': {'yolov8s':{'name':'YOLOv8s','available':True,'package':'Original six-class detector + balanced-crop classifiers','inference_size':640,'research':classifiers.research_summary()},
                         'yolo12':{'name':'YOLO12n','available':bool(active),'package':'Fold 1 demo export','inference_size':416,'research':yolo12_backend.research_summary() if active else None}},
            'techniques':classifiers.LABELS, 'technique_available': {key:key=='yolo' or (classifiers.ROOT/classifiers.FILES[key]).is_file() for key in classifiers.LABELS}, 'research': classifiers.research_summary(),
            'demos': json.loads(manifest.read_text()) if manifest.exists() else []}

def detect(image, confidence, iou, technique="yolo"):
    try: classifiers.validate(technique)
    except ValueError as exc: raise HTTPException(400,str(exc))
    if not 0.05 <= confidence <= 0.95 or not 0.1 <= iou <= 0.9:
        raise HTTPException(400, 'Threshold is outside the supported range.')
    started = time.perf_counter()
    with lock:
        detector=selected_model(technique)
        result = detector.predict(image, conf=confidence, iou=iou, imgsz=inference_size(technique), verbose=False)[0]
    detections = [{'class_id': int(b.cls.item()), 'class_name': detector.names[int(b.cls.item())],
                   'confidence': round(float(b.conf.item()), 4),
                   'box': [round(float(v), 1) for v in b.xyxy[0].tolist()]} for b in result.boxes]
    predictions=classifiers.classify(image,[d['box'] for d in detections],technique)
    for record,pred in zip(detections,predictions):
        record.update(detector_class=record['class_name'],detector_confidence=record['confidence'],classifier=technique if pred else None)
        if pred: record.update(pred)
    ok, encoded = cv2.imencode('.jpg', classifiers.annotate(result,detections), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(500, 'Could not render the detection result.')
    return {'image': 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode(),
            'detections': detections, 'elapsed_ms': round((time.perf_counter()-started)*1000),
            'width': image.shape[1], 'height': image.shape[0], 'technique': technique}

@app.post('/api/detect')
async def upload(file: UploadFile = File(...), confidence: float = Form(0.25), iou: float = Form(0.45), technique: str = Form("yolo")):
    content = await file.read(20 * 1024 * 1024 + 1)
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, 'Please upload an image smaller than 20 MB.')
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, 'Please select a valid JPG, PNG, or WebP image.')
    if image.shape[0] * image.shape[1] > 40_000_000:
        raise HTTPException(400, 'Please resize this image to under 40 megapixels.')
    return detect(image, confidence, iou, technique)

@app.post('/api/demo/{class_id}')
def demo(class_id: int, confidence: float = Form(0.25), iou: float = Form(0.45), technique: str = Form("yolo")):
    if class_id not in model.names:
        raise HTTPException(404, 'Unknown class.')
    path = ROOT / 'demos' / f'{class_id}.jpg'
    image = cv2.imread(str(path)) if path.exists() else None
    if image is None:
        raise HTTPException(404, 'Demo sample is unavailable.')
    return detect(image, confidence, iou, technique)

def process_video(job_id, source, confidence, iou, remove_source=False, technique="yolo"):
    cap = cv2.VideoCapture(str(source))
    writer = None
    raw = ROOT / 'results' / f'{job_id}_raw.mp4'
    output = ROOT / 'results' / f'{job_id}.mp4'
    started = time.perf_counter()
    try:
        classifiers.validate(technique)
        video_model = load_model('yolo12' if technique.startswith('yolo12_') else 'yolov8s')
        if not cap.isOpened():
            raise ValueError('Cannot read this video. Please use an MP4 file.')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        if not 1 <= fps <= 240:
            fps = 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total / fps > 120:
            raise ValueError('Please upload a video up to 2 minutes long.')
        frames = 0
        counts = {}
        peaks = {}
        trails = defaultdict(lambda: deque(maxlen=45))
        seen_tracks = set()
        while True:
            while jobs[job_id].get('paused') and not jobs[job_id].get('stop'):
                time.sleep(0.05)
            if jobs[job_id].get('stop'):
                break
            frame_started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            if frames >= int(fps * 120):
                break
            h, w = frame.shape[:2]
            if max(h, w) > 960:
                frame = cv2.resize(frame, (int(w*960/max(h,w)), int(h*960/max(h,w))))
            frame = frame[:frame.shape[0]//2*2, :frame.shape[1]//2*2]
            with lock:
                result = video_model.track(frame, persist=True, tracker='bytetrack.yaml', conf=confidence, iou=iou, imgsz=inference_size(technique), verbose=False)[0]
            present = set()
            live_tracks = []
            for box in result.boxes:
                name = video_model.names[int(box.cls.item())]
                present.add(name)
                peaks[name] = max(peaks.get(name, 0), float(box.conf.item()))
                track_id = int(box.id.item()) if box.id is not None else None
                coords = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                live_tracks.append({'track_id': track_id, 'class_name': name, 'confidence': round(float(box.conf.item()),4), 'box': coords})
                if track_id is not None:
                    seen_tracks.add(track_id)
                    trails[track_id].append((int((coords[0]+coords[2])/2), int((coords[1]+coords[3])/2)))
            predictions=classifiers.classify(frame,[t['box'] for t in live_tracks],technique)
            present=set()
            for record,pred in zip(live_tracks,predictions):
                record.update(detector_class=record['class_name'],detector_confidence=record['confidence'],classifier=technique if pred else None)
                if pred: record.update(pred)
                present.add(record['class_name'])
                peaks[record['class_name']]=max(peaks.get(record['class_name'],0),record['confidence'])
            for name in present:
                counts[name] = counts.get(name, 0) + 1
            if writer is None:
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise ValueError('Could not create the output video.')
            annotated = classifiers.annotate(result,live_tracks)
            for track in live_tracks:
                points = trails.get(track['track_id'], [])
                if len(points) > 1:
                    cv2.polylines(annotated, [np.array(points, np.int32)], False, (100,230,180), 2)
            writer.write(annotated)
            frames += 1
            encoded_ok, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if encoded_ok:
                jobs[job_id]['live'] = {'type': 'frame', 'technique': technique, 'frame': frames, 'image': 'data:image/jpeg;base64,'+base64.b64encode(jpeg).decode(), 'tracks': live_tracks, 'unique_tracks': len(seen_tracks), 'fps': round(1/max(time.perf_counter()-frame_started,0.001),1), 'video_seconds': round(frames/fps,2), 'progress': min(99,round(frames/max(total,1)*100))}
            jobs[job_id].update(progress=min(95, round(frames/max(total,1)*95)), frames=frames)
            time.sleep(max(0, 1/fps-(time.perf_counter()-frame_started)))
        if writer:
            writer.release()
            writer = None
        if not frames:
            raise ValueError('The video contains no readable frames.')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(raw), '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], check=True, capture_output=True)
        payload = {'status': 'complete', 'progress': 100, 'video': f'/results/{job_id}.mp4', 'frames': frames,
                   'technique': technique, 'stopped': bool(jobs[job_id].get('stop')), 'unique_tracks': len(seen_tracks),
                   'elapsed_ms': round((time.perf_counter()-started)*1000),
                   'detections': [{'class_name': name, 'frames_detected': count, 'confidence': round(peaks[name],4)} for name,count in counts.items()]}
        jobs[job_id].update(payload)
        (ROOT / 'results' / f'{job_id}.json').write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        jobs[job_id].update(status='failed', error=str(exc))
    finally:
        cap.release()
        if writer:
            writer.release()
        raw.unlink(missing_ok=True)
        if remove_source:
            source.unlink(missing_ok=True)

@app.post('/api/video')
async def video_upload(file: UploadFile = File(...), confidence: float = Form(0.25), iou: float = Form(0.45), technique: str = Form("yolo")):
    if not 0.05 <= confidence <= 0.95 or not 0.1 <= iou <= 0.9:
        raise HTTPException(400, 'Invalid threshold.')
    try: classifiers.validate(technique)
    except ValueError as exc: raise HTTPException(400,str(exc))
    job_id = uuid.uuid4().hex
    source = ROOT / 'results' / f'{job_id}_upload.mp4'
    size = 0
    with source.open('wb') as target:
        while content := await file.read(1024*1024):
            size += len(content)
            if size > 100*1024*1024:
                target.close()
                source.unlink(missing_ok=True)
                raise HTTPException(413, 'Please upload a video smaller than 100 MB.')
            target.write(content)
    jobs[job_id] = {'status': 'processing', 'progress': 0}
    pool.submit(process_video, job_id, source, confidence, iou, True, technique=technique)
    return {'job_id': job_id}

@app.post('/api/video-demo/{class_id}')
def video_demo(class_id: int, confidence: float = Form(0.25), iou: float = Form(0.45), technique: str = Form("yolo")):
    source = ROOT / 'demos' / f'{class_id}.mp4'
    if class_id not in model.names or not source.exists():
        raise HTTPException(404, 'Video demo unavailable.')
    if not 0.05 <= confidence <= 0.95 or not 0.1 <= iou <= 0.9:
        raise HTTPException(400, 'Invalid threshold.')
    try: classifiers.validate(technique)
    except ValueError as exc: raise HTTPException(400,str(exc))
    job_id = uuid.uuid4().hex
    jobs[job_id] = {'status': 'processing', 'progress': 0}
    pool.submit(process_video, job_id, source, confidence, iou, technique=technique)
    return {'job_id': job_id}

@app.get('/api/jobs/{job_id}')
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, 'Job not found.')
    return {k:v for k,v in jobs[job_id].items() if k != 'live'}

@app.post('/api/library/{clip_id}')
def library_demo(clip_id: str, confidence: float = Form(0.25), iou: float = Form(0.45), technique: str = Form("yolo")):
    manifest = json.loads((ROOT / 'demos/manifest.json').read_text())
    clip = next((clip for item in manifest for clip in item.get('clips',[]) if clip['id'] == clip_id), None)
    if clip is None:
        raise HTTPException(404, 'Demo clip not found.')
    source = (ROOT / 'demos' / clip['file']).resolve()
    if source.parent != (ROOT / 'demos').resolve() or not source.is_file():
        raise HTTPException(404, 'Demo file unavailable.')
    if not 0.05 <= confidence <= 0.95 or not 0.1 <= iou <= 0.9:
        raise HTTPException(400, 'Invalid threshold.')
    try: classifiers.validate(technique)
    except ValueError as exc: raise HTTPException(400,str(exc))
    job_id = uuid.uuid4().hex
    jobs[job_id] = {'status': 'processing', 'progress': 0}
    pool.submit(process_video, job_id, source, confidence, iou, technique=technique)
    return {'job_id': job_id}

@app.post('/api/jobs/{job_id}/{action}')
def control_job(job_id: str, action: str):
    if job_id not in jobs:
        raise HTTPException(404, 'Job not found.')
    if action not in ('pause', 'resume', 'stop'):
        raise HTTPException(400, 'Unknown action.')
    if action == 'stop':
        jobs[job_id]['stop'] = True
    else:
        jobs[job_id]['paused'] = action == 'pause'
    return {'ok': True}

@app.websocket('/api/live/{job_id}')
async def live_stream(socket: WebSocket, job_id: str):
    await socket.accept()
    if job_id not in jobs:
        await socket.send_json({'type':'error', 'error':'Job not found.'})
        await socket.close()
        return
    last_frame = 0
    try:
        while True:
            job = jobs[job_id]
            packet = job.get('live')
            if packet and packet['frame'] != last_frame:
                await socket.send_json(packet)
                last_frame = packet['frame']
            if job['status'] in ('complete','failed'):
                await socket.send_json({'type':'end', **{k:v for k,v in job.items() if k != 'live'}})
                break
            await asyncio.sleep(0.03)
    except WebSocketDisconnect:
        pass


@app.websocket('/api/webcam')
async def webcam(socket: WebSocket):
    await socket.accept()
    try:
        settings=await socket.receive_json()
        technique=settings.get('technique','yolo')
        classifiers.validate(technique)
        confidence=float(settings.get('confidence',.25));iou=float(settings.get('iou',.45))
        if not .05<=confidence<=.95 or not .1<=iou<=.9:raise ValueError('Invalid threshold.')
        detector=await asyncio.to_thread(load_model,'yolo12' if technique.startswith('yolo12_') else 'yolov8s')
        frame_id=0;seen=set()
        def infer_frame(image):
            started=time.perf_counter()
            with lock:
                result=detector.track(image,persist=True,tracker='bytetrack.yaml',conf=confidence,iou=iou,imgsz=inference_size(technique),verbose=False)[0]
            records=[]
            for box in result.boxes:
                records.append({'track_id':int(box.id.item()) if box.id is not None else None,
                    'class_name':detector.names[int(box.cls.item())],'confidence':round(float(box.conf.item()),4),
                    'box':[round(float(v),1) for v in box.xyxy[0].tolist()]})
            for record,pred in zip(records,classifiers.classify(image,[r['box'] for r in records],technique)):
                record.update(detector_class=record['class_name'],detector_confidence=record['confidence'],classifier=technique if pred else None)
                if pred:record.update(pred)
                if record['track_id'] is not None:seen.add(record['track_id'])
            ok,jpg=cv2.imencode('.jpg',classifiers.annotate(result,records),[cv2.IMWRITE_JPEG_QUALITY,80])
            if not ok:raise ValueError('Could not render frame.')
            return {'type':'frame','technique':technique,'frame':frame_id,'tracks':records,'unique_tracks':len(seen),
                    'image':'data:image/jpeg;base64,'+base64.b64encode(jpg).decode(),'fps':round(1/max(.001,time.perf_counter()-started),1),
                    'video_seconds':0,'progress':0}
        while True:
            content=await socket.receive_bytes()
            if len(content)>2*1024*1024:raise ValueError('Camera frame too large.')
            image=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
            if image is None or max(image.shape[:2])>1280:raise ValueError('Invalid camera frame.')
            frame_id+=1
            await socket.send_json(await asyncio.to_thread(infer_frame,image))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await socket.send_json({'type':'error','error':str(exc)})
        await socket.close()
