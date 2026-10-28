let cameraStream=null,cameraSocket=null,cameraVideo=document.createElement('video');
cameraVideo.muted=true;cameraVideo.playsInline=true;
function stopCamera(){
 if(cameraSocket)cameraSocket.close();cameraSocket=null;
 if(cameraStream)cameraStream.getTracks().forEach(t=>t.stop());cameraStream=null;
 cameraVideo.srcObject=null;busy=false;
 document.querySelectorAll('button').forEach(b=>b.disabled=false);
 $('family').disabled=false;$('technique').disabled=false;$('file').disabled=false;
 $('camera-stop').hidden=true;$('run').disabled=!chosen;
}
$('camera-stop').onclick=()=>{stopCamera();$('status').textContent='Camera stopped';};
$('camera-start').onclick=async()=>{
 if(busy)return;
 clearResult();busy=true;
 document.querySelectorAll('button').forEach(b=>b.disabled=true);
 $('family').disabled=true;$('technique').disabled=true;$('file').disabled=true;
 try{
  cameraStream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960},height:{ideal:540}},audio:false});
  cameraVideo.srcObject=cameraStream;await cameraVideo.play();
  const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');
  const ratio=Math.min(1,960/Math.max(cameraVideo.videoWidth,cameraVideo.videoHeight));
  canvas.width=Math.round(cameraVideo.videoWidth*ratio);canvas.height=Math.round(cameraVideo.videoHeight*ratio);
  const ws=cameraSocket=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/api/webcam');
  function send(){if(cameraSocket!==ws||ws.readyState!==WebSocket.OPEN)return;ctx.drawImage(cameraVideo,0,0,canvas.width,canvas.height);canvas.toBlob(blob=>{if(blob&&ws.readyState===WebSocket.OPEN)ws.send(blob);},'image/jpeg',.8);}
  ws.onopen=()=>{ws.send(JSON.stringify({technique:selectedTechnique(),confidence:$('conf').value/100,iou:$('iou').value/100}));send();};
  ws.onmessage=e=>{let p=JSON.parse(e.data);if(p.type==='error'){showError(p.error);stopCamera();return;}drawLive(p);$('status').textContent='LIVE CAMERA · '+$('technique').selectedOptions[0].text+' · '+p.fps+' FPS';send();};
  ws.onerror=()=>{showError('Camera detection connection failed.');stopCamera();};
  ws.onclose=()=>{if(cameraSocket===ws)stopCamera();};
  $('camera-stop').hidden=false;$('camera-stop').disabled=false;
 }catch(e){stopCamera();showError('Cannot start camera: '+e.message);}
};
window.addEventListener('pagehide',stopCamera);
