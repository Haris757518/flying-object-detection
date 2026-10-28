import json
from pathlib import Path
from fastapi.testclient import TestClient
import app

root = Path(__file__).resolve().parent
client = TestClient(app.app)
assert client.get('/').status_code == 200
assert len(client.get('/api/model').json()['classes']) == 6
report = {'images': [], 'videos': []}
for cid, name in app.model.names.items():
    response = client.post(f'/api/demo/{cid}')
    assert response.status_code == 200, response.text
    result = response.json()
    assert any(d['class_id'] == cid for d in result['detections']), name
    report['images'].append({'class': name, 'passed': True})
    job_id = f'verify_{cid}'
    app.jobs[job_id] = {'status': 'processing', 'progress': 0}
    app.process_video(job_id, root/'demos'/f'{cid}.mp4', .25, .45)
    result = app.jobs[job_id]
    assert result['status'] == 'complete', result
    assert any(d['class_name'] == name for d in result['detections']), (name, result)
    report['videos'].append({'class': name, **result})
    print(name, [(d['class_name'],d['frames_detected']) for d in result['detections']], flush=True)
assert client.post('/api/detect', files={'file': ('bad.jpg', b'bad', 'image/jpeg')}).status_code == 400
assert client.post('/api/demo/99').status_code == 404
(root/'verification.json').write_text(json.dumps(report, indent=2))
print('Verification completed', flush=True)
