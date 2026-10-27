'use strict';
const $ = id => document.getElementById(id);
const deployment = window.CARTPOLE_CONFIG || {};
const apiOrigin = deployment.apiOrigin || '';
const apiURL = path => apiOrigin + path;
let token = sessionStorage.getItem('cartpole-token') || '';
let localCredentials = null, approvalJob = null, identity = null;
let jobs = [], reservations = [], config = {}, activeView = 'experiments';
let videoPlayer = null;
let cameraSource = 'arducam';
function cameraAvailable() { return config.camera_sources ? config.camera_sources.some(s => s.id === cameraSource) : Boolean(config.camera_url && config.camera_available !== false); }
function configureCameraSources() {
  const select = $('camera-source-select');
  const sources = config.camera_sources || (config.camera_url && config.camera_available !== false ? [{id:'arducam',label:'Arducam'}] : []);
  if (!sources.some(s => s.id === cameraSource)) cameraSource = sources[0]?.id || 'arducam';
  select.replaceChildren();
  for (const source of sources) { const option = document.createElement('option'); option.value = source.id; option.textContent = source.label; select.append(option); }
  select.value = cameraSource; select.disabled = sources.length < 2;
}
let cameraBlob = null, cameraAbort = null;
let cameraRunning = false, cameraTimer = null, cameraWatchdog = null, cameraGeneration = 0;
$('token').value = token;

async function request(path, body) {
  const response = await fetch(apiURL(path), {
    redirect: 'error',
    method: body === undefined ? 'GET' : 'POST',
    headers: { Authorization: 'Bearer ' + token, ...(body === undefined ? {} : {'Content-Type': 'application/json'}) },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || `Request failed (${response.status})`);
  }
  return response;
}
async function api(path, body) { return (await request(path, body)).json(); }
function message(text) { $('message').textContent = text; }
function action(label, handler, quiet = true) {
  const button = document.createElement('button');
  button.type = 'button'; button.textContent = label;
  if (quiet) button.className = 'quiet';
  button.onclick = handler;
  return button;
}
function showView(view) {
  if (!['experiments', 'docs', 'data'].includes(view)) view = 'experiments';
  activeView = view;
  for (const panel of document.querySelectorAll('.view')) panel.hidden = panel.id !== 'view-' + view;
  for (const button of document.querySelectorAll('[data-view]')) {
    const selected = button.dataset.view === view;
    button.classList.toggle('active', selected);
    if (selected) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
  }
  if (view !== 'experiments') pauseCamera();
  else autoStartCamera();
  if (view === 'reservations' && token) refreshReservations();
}
for (const button of document.querySelectorAll('[data-view]')) button.onclick = () => { location.hash = button.dataset.view; };
window.addEventListener('hashchange', () => showView(location.hash.slice(1)));
showView(location.hash.slice(1));
$('access-toggle').onclick = () => { $('access-panel').open = true; $('access-panel').scrollIntoView({block: 'center'}); $('token').focus(); };

function artifactFormat(filename) {
  const extension = filename.toLowerCase().split('.').pop();
  return ({py: 'Python controller', zip: 'Python project', json: 'Trained actor'})[extension] || null;
}
async function submissionArtifact(file) {
  if (!file || !artifactFormat(file.name)) throw new Error('Choose a Python file, ZIP project, or exported actor JSON.');
  const python = file.name.toLowerCase().endsWith('.py');
  const actor = file.name.toLowerCase().endsWith('.json');
  const limit = python ? 65536 : actor ? 1000000 : 8 * 1024 * 1024;
  if (file.size < 1 || file.size > limit) throw new Error(python ? 'Python files must be nonempty and at most 64 KiB.' : actor ? 'Actor JSON must be nonempty and at most 1,000,000 bytes (1 MB).' : 'ZIP projects must be nonempty and at most 8 MiB.');
  if (python) return {source: await file.text()};
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += 16384) binary += String.fromCharCode(...bytes.subarray(offset, offset + 16384));
  return {artifact_base64: btoa(binary)};
}
function artifactInspection(job, artifact, bytes) {
  const heading = `${artifactFormat(job.filename) || 'Artifact'}: ${job.filename}\nSHA-256: ${artifact.sha256}\nSize: ${bytes.length.toLocaleString()} bytes`;
  if (job.filename.toLowerCase().endsWith('.py')) return heading + '\n\n' + new TextDecoder('utf-8', {fatal:true}).decode(bytes);
  const metadata = artifact.inspection || artifact.metadata;
  return heading + (metadata ? '\n\n' + JSON.stringify(metadata, null, 2) : '\n\nDownload the submitted artifact to inspect its contents.');
}
function offerArtifactDownload(job, bytes) {
  const button = $('artifact-download'); button.hidden = false;
  button.onclick = () => {
    const url = URL.createObjectURL(new Blob([bytes], {type:'application/octet-stream'}));
    const link = document.createElement('a'); link.href = url; link.download = job.filename;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  };
}
function runRow(job) {
  const row = document.createElement('article'); row.className = 'run';
  const title = document.createElement('div'); title.className = 'run-title';
  const filename = document.createElement('strong'); filename.textContent = job.filename;
  const status = document.createElement('span'); status.className = 'status'; status.dataset.state = job.status;
  status.textContent = job.status.replaceAll('_', ' ');
  title.append(filename, status);
  const detail = document.createElement('small');
  detail.textContent = `${job.duration} seconds · ${job.source === 'simulation' ? 'Simulation' : 'Physical rig'} · ${new Date(job.created * 1000).toLocaleString()}`;
  row.append(title, detail);
  return row;
}
function hasRecording(job) { return job.result?.recording?.status === 'available'; }
async function downloadRecording(job, button) {
  button.disabled = true;
  try {
    const response = await request('/api/jobs/' + encodeURIComponent(job.id) + '/recording');
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a'); link.href = url; link.download = 'cartpole-run-' + job.id + '.zip';
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  } catch (error) { message(error.message); }
  finally { button.disabled = false; }
}
function downloadButton(job) {
  const button = action('Download bundle', () => downloadRecording(job, button));
  return button;
}
function renderJobs() {
  $('jobs').replaceChildren(); $('data-jobs').replaceChildren();
  $('run-count').textContent = String(jobs.length); $('empty').hidden = jobs.length > 0;
  $('data-empty').hidden = jobs.length > 0;
  for (const job of jobs) {
    const row = runRow(job), actions = document.createElement('div'); actions.className = 'run-actions';
    actions.append(action('View details', () => { $('artifact-download').hidden = true; $('artifact-download').onclick = null; $('result-body').textContent = JSON.stringify(job, null, 2); $('result').open = true; $('result').scrollIntoView({block: 'nearest'}); }));
    if (hasRecording(job)) actions.append(downloadButton(job));
    if (identity?.role === 'submit' && ['awaiting_approval', 'pending', 'claimed', 'running'].includes(job.status)) {
      const cancel = action(job.cancel_requested ? 'Stop requested' : 'Cancel run', async () => {
        cancel.disabled = true;
        try { await api('/api/jobs/' + job.id + '/cancel', {}); await refresh(); }
        catch (error) { message(error.message); cancel.disabled = false; }
      }); cancel.disabled = job.cancel_requested; actions.append(cancel);
    }
    if (identity?.role === 'operator' && job.status === 'awaiting_approval') {
      actions.append(action('Inspect artifact', async () => {
        try {
          const artifact = await api('/api/jobs/' + job.id + '/artifact');
          if (artifact.sha256 !== job.artifact) throw new Error('Artifact identity changed');
          const bytes = Uint8Array.from(atob(artifact.artifact_base64 || artifact.source_base64), c => c.charCodeAt(0));
          $('result-body').textContent = artifactInspection(job, artifact, bytes);
          offerArtifactDownload(job, bytes);
          $('result').open = true; $('result').scrollIntoView({block: 'nearest'});
        } catch (error) { message(error.message); }
      }), action('Approve physical run', () => openApproval(job), false));
    }
    row.append(actions); $('jobs').append(row);
    const dataRow = runRow(job), dataActions = document.createElement('div'); dataActions.className = 'run-actions';
    if (hasRecording(job)) dataActions.append(downloadButton(job));
    else { const note = document.createElement('small'); note.textContent = ['completed', 'failed', 'cancelled'].includes(job.status) ? 'No recorded bundle is attached to this run.' : 'Recording becomes available after execution.'; dataActions.append(note); }
    dataRow.append(dataActions); $('data-jobs').append(dataRow);
  }
}
async function refresh() {
  try {
    identity = await api('/api/me');
    const data = await api('/api/jobs'); jobs = data.jobs;
    $('submission').hidden = identity.role !== 'submit';
    $('operator-upload-note').hidden = identity.role === 'submit';
    $('connection').textContent = identity.role + ' · ' + identity.principal;
    document.body.classList.add('connected');
    $('access-panel').open = false;
    renderJobs(); message('');
    try {
      config = await api('/api/config');
      $('recording').disabled = !config.recording_modes?.includes('detailed');
      $('recording-status').textContent = $('recording').disabled ? 'This backend supports its default trace only; detailed recording requires an API and worker update.' : 'Detailed collection is bounded and depends on the worker’s configured sources.';
      configureCameraSources();
      $('camera-source').textContent = !cameraAvailable() ? 'No camera source configured.' : (apiOrigin || location.origin) + ' · authenticated camera';
      $('camera-start').disabled = !cameraAvailable();
      autoStartCamera();
      await refreshCameraMode();
      await refreshDemos();
      pollAngles();
    } catch (error) {
      config = {}; $('recording').disabled = true; $('camera-source').textContent = 'Camera configuration is unavailable. Restart the updated portal server.';
      $('camera-start').disabled = true;
      $('reservation-status').textContent = 'Reservation configuration is unavailable. Restart the updated portal server.';
      $('reservation-form').hidden = true;
    }
  } catch (error) { $('connection').textContent = 'Connection needs attention'; document.body.classList.remove('connected'); message(error.message); }
}
$('connect').onclick = () => { token = $('token').value.trim(); sessionStorage.setItem('cartpole-token', token); refresh(); };
$('forget').onclick = () => {
  pauseCamera(); clearTimeout(angleTimer); for(const id of ['controller','policy']) $('demo-'+id).disabled=true; localCredentials = null; identity = null; config = {}; jobs = []; reservations = [];
  $('credential-file').value = ''; $('dev-role').disabled = true; token = ''; $('token').value = '';
  sessionStorage.removeItem('cartpole-token'); renderJobs(); $('reservations').replaceChildren();
  $('connection').textContent = 'Not connected'; document.body.classList.remove('connected');
  $('artifact-download').hidden = true; $('artifact-download').onclick = null;
  $('result-body').textContent = 'Select a run to inspect its result.'; $('result').open = false;
  $('reservation-form').hidden = true; $('reservation-status').textContent = 'Connect to check reservation availability.';
  $('reservation-assignment').hidden = true; $('camera-source').textContent = 'Connect to load the camera configuration.';
  $('camera-start').disabled = true; $('camera-mode').disabled = true; $('submission').hidden = false; $('operator-upload-note').hidden = true;
  message(''); $('access-panel').open = true;
};
$('refresh').onclick = refresh; $('refresh-data').onclick = refresh;
$('file').onchange = () => {
  const file = $('file').files[0];
  $('file-label').textContent = file?.name || 'Choose a controller or policy';
  $('file-kind').textContent = file ? artifactFormat(file.name) || 'Unsupported file' : '.py / .zip / .json';
};
$('source').onchange = updateReservationAssignment;
$('submission').onsubmit = async event => {
  event.preventDefault(); const file = $('file').files[0];
  $('upload').disabled = true;
  try {
    const artifact = await submissionArtifact(file);
    const body = {...artifact, filename: file.name, sensor_source: $('source').value, duration: Number($('duration').value), idempotency_key: crypto.randomUUID()};
    if (config.recording_modes?.includes($('recording').value)) body.recording = $('recording').value;
    if (body.sensor_source !== 'simulation' && $('run-reservation').value) body.reservation_id = $('run-reservation').value;
    await api('/api/jobs', body); await refresh(); message('Experiment queued.');
  } catch (error) { message(error.message); }
  finally { $('upload').disabled = false; }
};
function openApproval(job) {
  approvalJob = job; $('approval-detail').textContent = `${job.filename} · ${job.duration} seconds · ${job.source}\nSHA-256: ${job.artifact}`;
  $('approval-error').textContent = ''; $('approval-dialog').showModal();
}
$('approval-cancel').onclick = () => $('approval-dialog').close();
$('approval-dialog').addEventListener('close', () => { approvalJob = null; });
$('approval-confirm').onclick = async () => {
  if (!approvalJob) return; const job = approvalJob; $('approval-confirm').disabled = true;
  try { await api('/api/jobs/' + job.id + '/approve', {artifact: job.artifact, sensor_source: job.source, duration: job.duration}); $('approval-dialog').close(); await refresh(); }
  catch (error) { $('approval-error').textContent = error.message; }
  finally { $('approval-confirm').disabled = false; }
};
function connectLocalRole() {
  const selected = Object.entries(localCredentials || {}).find(([, principal]) => principal[1] === $('dev-role').value);
  if (!selected) { message('No credential for this role.'); return; }
  $('token').value = selected[0]; $('connect').click();
}
$('dev-role').onchange = connectLocalRole;
$('credential-file').onchange = async () => {
  const file = $('credential-file').files[0]; if (!file) return;
  try {
    if (file.size > 16384) throw new Error('Credential file is too large.');
    const data = JSON.parse(await file.text());
    const entries = data && !Array.isArray(data) && typeof data === 'object' ? Object.entries(data) : [];
    if (!entries.length || entries.length > 2 || !entries.every(([key, value]) =>
      /^[0-9a-f]{64}$/.test(key) && Array.isArray(value) && value.length === 2 &&
      typeof value[0] === 'string' && value[0].length > 0 && ['submit','operator'].includes(value[1])) ||
      new Set(entries.map(([,value]) => value[1])).size !== entries.length)
      throw new Error('Choose a portal sign-in file containing a submitter or operator credential.');
    pauseCamera();
    localCredentials = data;
    for (const option of $('dev-role').options) option.disabled = !entries.some(([,value]) => value[1] === option.value);
    $('dev-role').value = entries.some(([,value]) => value[1] === 'submit') ? 'submit' : 'operator';
    $('dev-role').disabled = entries.length === 1;
    connectLocalRole();
  } catch (error) { message(error.message); }
};

function pauseCamera() {
  videoPlayer?.stop(); cameraAbort?.abort(); cameraAbort = null;
  if (cameraBlob) URL.revokeObjectURL(cameraBlob); cameraBlob = null;
  $('camera-video').hidden = true;
  cameraRunning = false; cameraGeneration++; clearTimeout(cameraTimer); clearTimeout(cameraWatchdog);
  $('camera-preview').onload = null; $('camera-preview').onerror = null;
  $('camera-preview').removeAttribute('src'); $('camera-preview').hidden = true;
  $('camera-placeholder').hidden = false; $('camera-status').textContent = 'Camera paused';
}
async function cameraFrame(generation) {
  if (!cameraRunning || generation !== cameraGeneration || activeView !== 'experiments' || document.hidden) return;
  const image = $('camera-preview');
  let frameFailed = false;
  const failed = () => {
    if (generation !== cameraGeneration || frameFailed) return;
    frameFailed = true;
    clearTimeout(cameraWatchdog); image.hidden = true; $('camera-placeholder').hidden = false;
    $('camera-status').textContent = 'Camera unavailable';
    $('camera-error').textContent = 'Cannot reach the camera preview. Check the camera service, tunnel, and configured source.';
    cameraTimer = setTimeout(() => cameraFrame(generation), 3000);
  };
  image.onload = () => {
    if (generation !== cameraGeneration) return;
    clearTimeout(cameraWatchdog); image.hidden = false; $('camera-placeholder').hidden = true;
    $('camera-status').textContent = 'Sampled JPEG preview';
    cameraTimer = setTimeout(() => cameraFrame(generation), 250);
  };
  image.onerror = failed;
  cameraAbort = new AbortController();
  cameraWatchdog = setTimeout(() => { cameraAbort?.abort(); failed(); }, 15000);
  try {
    const response = await fetch(apiURL(cameraSource === 'emeet' ? '/api/camera/preview?source=emeet' : '/api/camera/preview?mode=' + encodeURIComponent($('camera-mode').value)), {
      headers: {Authorization: 'Bearer ' + token}, cache: 'no-store',
      redirect: 'error', signal: cameraAbort.signal,
    });
    if (!response.ok) throw new Error('Camera preview unavailable');
    const blob = await response.blob();
    if (generation !== cameraGeneration) return;
    if (cameraBlob) URL.revokeObjectURL(cameraBlob);
    cameraBlob = URL.createObjectURL(blob); image.src = cameraBlob;
  } catch (error) { if (generation === cameraGeneration) failed(); }

}
function autoStartCamera() {
  if (activeView === 'experiments' && !document.hidden && identity && token && cameraAvailable() && !cameraRunning) startCamera();
}
function startCamera() {
  pauseCamera();
  try {
    if (!cameraAvailable()) throw new Error('The operator must configure a camera source.');
    cameraRunning = true; $('camera-error').textContent = '';
    if (!videoPlayer) videoPlayer = new PortalVideo($('camera-video'), {
      status: status => { $('camera-status').textContent = status; $('camera-video').hidden = true; $('camera-placeholder').hidden = false; },
      frame: () => { $('camera-video').hidden = false; $('camera-preview').hidden = true; $('camera-placeholder').hidden = true; $('camera-status').textContent = 'Live video'; $('camera-error').textContent = ''; },
      fallback: reason => { if (!cameraRunning) return; $('camera-video').hidden = true; $('camera-error').textContent = reason; cameraFrame(cameraGeneration); },
    });
    videoPlayer.start(token, apiURL(cameraSource === 'emeet' ? '/api/video?source=emeet' : '/api/video'), cameraSource === 'emeet' ? 'mjpeg' : 'h264');
  } catch (error) { $('camera-error').textContent = error.message; }
};
$('camera-start').onclick = startCamera;
$('camera-stop').onclick = pauseCamera;
document.addEventListener('visibilitychange', () => { if (document.hidden) pauseCamera(); });

function updateReservationAssignment() {
  const select = $('run-reservation'), selected = select.value;
  select.replaceChildren(new Option('No reservation', ''));
  for (const reservation of reservations) {
    if (reservation.owner === identity?.principal && reservation.status !== 'cancelled' && reservation.ends_at > Date.now() / 1000) select.add(new Option(new Date(reservation.starts_at * 1000).toLocaleString(), reservation.id));
  }
  if ([...select.options].some(option => option.value === selected)) select.value = selected;
  $('reservation-assignment').hidden = true;
}
async function refreshReservations() {
  try {
    const data = await api('/api/reservations'); reservations = data.reservations;
    $('reservation-form').hidden = !data.enabled || identity?.role !== 'submit';
    $('reservation-status').textContent = !data.enabled ? 'Reservations are not enabled on this deployment. You can still submit individual runs.' : identity?.role !== 'submit' ? 'Reservations are enabled. Use a submitter credential to reserve a session.' : 'Reserve a 30-minute window. Overlapping sessions are not available.';
    $('reservations').replaceChildren();
    for (const reservation of reservations) {
      const row = document.createElement('article'); row.className = 'run';
      const title = document.createElement('strong'); title.textContent = new Date(reservation.starts_at * 1000).toLocaleString();
      const details = document.createElement('small'); details.textContent = `${Math.round((reservation.ends_at - reservation.starts_at) / 60)} minutes · ${reservation.status} · ${reservation.owner}`;
      row.append(title, details);
      if (reservation.status !== 'cancelled' && reservation.ends_at > Date.now() / 1000 && (identity?.role === 'submit' && reservation.owner === identity?.principal)) {
        const actions = document.createElement('div'); actions.className = 'run-actions';
        actions.append(action('Cancel reservation', async () => { try { await api('/api/reservations/' + reservation.id + '/cancel', {}); await refreshReservations(); } catch (error) { $('reservation-message').textContent = error.message; } })); row.append(actions);
      }
      $('reservations').append(row);
    }
    if (!reservations.length) { const empty = document.createElement('p'); empty.className = 'muted'; empty.textContent = 'No sessions reserved yet.'; $('reservations').append(empty); }
    updateReservationAssignment();
  } catch (error) { $('reservation-status').textContent = error.message; $('reservation-form').hidden = true; }
}
$('refresh-reservations').onclick = refreshReservations;
const nextStart = new Date(Date.now() + 60000); nextStart.setSeconds(0, 0);
$('reservation-start').value = new Date(nextStart.getTime() - nextStart.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
$('reservation-form').onsubmit = async event => {
  event.preventDefault(); const button = event.submitter; button.disabled = true;
  try {
    await api('/api/reservations', {rig: config.rig_id || 'cartpole', starts_at: new Date($('reservation-start').value).getTime() / 1000, duration_minutes: Number($('reservation-minutes').value)});
    $('reservation-message').textContent = 'Session reserved.'; await refreshReservations();
  } catch (error) { $('reservation-message').textContent = error.message; }
  finally { button.disabled = false; }
};
if (!deployment.hosted) fetch(apiURL('/api/dev-info'), {redirect: 'error'}).then(response => response.ok ? response.json() : null).then(info => {
  if (!deployment.hosted && info?.local_preview === true && ['127.0.0.1', 'localhost'].includes(location.hostname)) {
    $('dev-preview').hidden = false; $('dev-access').hidden = false; $('environment').textContent = 'Local development';
  }
}).catch(() => {});
$('camera-start').disabled = true;
if (token) refresh();

async function refreshCameraMode() {
  const generation = cameraGeneration;
  const select = $('camera-mode'); select.disabled = true;
  $('camera-mode-control').hidden = cameraSource === 'emeet';
  if (cameraSource === 'emeet') { $('camera-mode-note').textContent = 'EMEET webcam · MJPEG preview. Tracking and mask belong to Arducam.'; return; }
  try {
    const state = await api('/api/camera');
    if (generation !== cameraGeneration || cameraSource !== 'arducam') return;
    for (const option of select.options) option.disabled = !state.modes.includes(option.value);
    select.value = state.mode; select.disabled = false;
    $('camera-mode-note').textContent = 'Presentation is shared with other viewers. This changes the video view only.';
  } catch (error) {
    if (generation !== cameraGeneration || cameraSource !== 'arducam') return;
    $('camera-mode-note').textContent = 'Camera presentation controls are unavailable: ' + error.message;
  }
}
$('camera-mode').onchange = async () => {
  const select = $('camera-mode'); select.disabled = true;
  try {
    await api('/api/camera/mode', {mode: select.value});
    if (cameraRunning) $('camera-start').click();
  } catch (error) { $('camera-error').textContent = error.message; }
  finally { await refreshCameraMode(); }
};

$('camera-source-select').onchange = async () => {
  const running = cameraRunning; pauseCamera();
  const generation = cameraGeneration;
  cameraSource = $('camera-source-select').value === 'emeet' ? 'emeet' : 'arducam';
  $('camera-start').disabled = !cameraAvailable();
  await refreshCameraMode();
  if (running && generation === cameraGeneration && activeView === 'experiments' && !document.hidden) startCamera();
};


async function refreshDemos() {
  try {
    const data = await api('/api/demos');
    for (const id of ['controller','policy']) $('demo-' + id).disabled = !data.demos.some(d => d.id === id && d.available);
    $('demo-status').textContent = 'Physical demos require operator approval. No motion starts on click.';
  } catch (error) {
    for (const id of ['controller','policy']) $('demo-' + id).disabled = true;
    $('demo-status').textContent = 'Demos unavailable: ' + error.message;
  }
}
for (const id of ['controller','policy']) $('demo-' + id).onclick = async () => {
  const button = $('demo-' + id); button.disabled = true;
  try {
    const job = await api('/api/demos/' + id + '/run', {idempotency_key: 'demo-' + id + '-' + Date.now() + '-' + Math.random().toString(36).slice(2)});
    await refresh();
    $('demo-status').textContent = 'Demo queued · ' + job.id + ' · awaiting operator approval.';
  } catch (error) { $('demo-status').textContent = error.message; }
  finally { button.disabled = !identity; }
};
const angleDisplay = {vision:{},encoder:{},now:null};
function drawAngle(id, source, now) {
  angleDisplay[id]=source;angleDisplay.now=now;
  const rows=source.samples||[],last=rows.at(-1);
  const stale=source.stale || !last || !Number.isFinite(last.theta_rad) || !Number.isFinite(now) || now-last.t>.5 || last.angle_valid===false;
  $(id+'-angle-value').textContent=stale?'No live data':angleDegrees(last.theta_rad).toFixed(1)+'°';
  $(id+'-angle-status').textContent=(id==='vision'?'Vision: ':'Encoder: ')+(stale?'missing or stale measurements.':'live · upright = 0°.');
  renderAngleChart();
}
function angleDegrees(q) {return ((q*180/Math.PI+180)%360+360)%360-180;}
function renderAngleChart() {
  const canvas=$('angle-chart'),ctx=canvas.getContext?.('2d');if(!ctx)return;
  const w=canvas.width,h=canvas.height,left=42,right=w-16,top=16,bottom=h-28,now=angleDisplay.now;
  ctx.clearRect(0,0,w,h);ctx.font='13px sans-serif';ctx.lineWidth=1;
  for(const angle of [-180,-90,0,90,180]) {const y=top+(180-angle)/360*(bottom-top);ctx.strokeStyle='#dce4ed';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillStyle='#65758a';ctx.fillText(angle+'°',0,y+4);}
  ctx.fillText('−15 s',left,h-5);ctx.fillText('now',right-24,h-5);
  if(!Number.isFinite(now))return;
  for(const id of ['vision','encoder']) {
    if($(id+'-angle-toggle').checked===false)continue;
    ctx.strokeStyle=id==='vision'?'#24786f':'#315dc5';ctx.lineWidth=2;ctx.setLineDash(id==='encoder'?[6,3]:[]);ctx.beginPath();let previous=null;
    for(const row of angleDisplay[id].samples||[]) {
      if(!Number.isFinite(row.t)||!Number.isFinite(row.theta_rad)||row.t<now-15||row.t>now||row.angle_valid===false){previous=null;continue;}
      const angle=angleDegrees(row.theta_rad),x=left+(row.t-(now-15))/15*(right-left),y=top+(180-angle)/360*(bottom-top);
      if(!previous||row.gap||row.t-previous.t>.25||Math.abs(angle-previous.angle)>180)ctx.moveTo(x,y);else ctx.lineTo(x,y);
      previous={t:row.t,angle};
    }ctx.stroke();
  }ctx.setLineDash([]);
}
for(const id of ['vision','encoder'])$(id+'-angle-toggle').onchange=renderAngleChart;
let angleTimer=null, anglePolling=false;
async function pollAngles() {
  clearTimeout(angleTimer);
  if(anglePolling || activeView!=='experiments' || document.hidden || !token || !identity) return;
  anglePolling=true;
  try { const data=await api('/api/angles'); for(const id of ['vision','encoder'])drawAngle(id,data[id]||{},data.now); }
  catch(error) {for(const id of ['vision','encoder'])drawAngle(id,{},null);}
  finally {anglePolling=false;angleTimer=setTimeout(pollAngles,1000);}
}
window.addEventListener('hashchange', pollAngles);
document.addEventListener('visibilitychange', pollAngles);
