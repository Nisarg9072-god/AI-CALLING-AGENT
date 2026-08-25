/**
 * AI Calling Agent — Browser Voice Client
 *
 * Flow:
 *   1. User clicks Start Call → creates session via POST /calls
 *   2. WebSocket connects to /ws/voice/{session_id}
 *   3. User holds "Hold to Speak" → getUserMedia → AudioContext captures PCM
 *   4. On release → sends audio_end → backend runs STT → Agent → TTS
 *   5. Backend sends back base64 PCM audio → browser plays it
 *   6. Agent events shown in real-time trace panel
 */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────────

const state = {
  ws: null,
  sessionId: null,
  callId: null,
  connected: false,
  recording: false,
  audioCtx: null,
  mediaStream: null,
  recorder: null,
  audioChunks: [],
  currentIteration: 0,
  iterationGroups: {},    // { iteration: traceGroupEl }
  callStartTime: null,
  sttLatencyMs: null,
  totalLatencyMs: null,
  turnCount: 0,
};

// ── Audio playback queue ───────────────────────────────────────────────────────

const audioQueue = {
  queue: [],
  playing: false,

  enqueue(pcm16Bytes, sampleRate) {
    this.queue.push({ pcm16Bytes, sampleRate });
    if (!this.playing) this._playNext();
  },

  async _playNext() {
    if (this.queue.length === 0) { this.playing = false; return; }
    this.playing = true;
    const { pcm16Bytes, sampleRate } = this.queue.shift();
    try {
      await playPCM16(pcm16Bytes, sampleRate);
    } catch (e) {
      console.warn('Audio playback error:', e);
    }
    this._playNext();
  }
};

// ── DOM helpers ───────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

function setStatus(text, mode) {
  const pill = $('connection-status');
  pill.querySelector('.status-text').textContent = text;
  pill.className = 'status-pill';
  if (mode) pill.classList.add(mode);
}

function setMicStatus(label, detail, active = false) {
  $('mic-label').textContent = label;
  $('mic-detail').textContent = detail;
  $('mic-icon').className = 'mic-icon' + (active ? ' active' : '');
}

function showAction(text) {
  const el = $('current-action');
  el.style.display = 'flex';
  $('current-action-text').textContent = text;
}

function hideAction() {
  $('current-action').style.display = 'none';
}

function setMicLevel(level) {
  $('mic-level').style.width = Math.min(100, level * 100) + '%';
}

// ── Session management ────────────────────────────────────────────────────────

async function startCall() {
  const phone = $('phone-input').value.trim() || '+919537266092';

  try {
    setStatus('Creating session...', '');
    showAction('Creating call session...');

    const res = await fetch('/calls', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phone_number: phone, transport: 'websocket' }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to create session');
    }

    const data = await res.json();
    state.sessionId = data.session_id;
    state.callId = data.call_id;

    // Connect WebSocket
    const wsUrl = `ws://${location.host}/ws/voice/${state.sessionId}`;
    connectWebSocket(wsUrl);

    // Enable mic
    await initMicrophone();

    $('start-btn').disabled = true;
    $('end-btn').disabled = false;
    $('record-btn').disabled = false;
    $('text-section').style.display = 'block';
    state.callStartTime = Date.now();

    clearTranscript();
    clearTrace();

  } catch (err) {
    setStatus('Error', 'error');
    hideAction();
    alert('Failed to start call: ' + err.message);
  }
}

function endCall() {
  if (state.ws) {
    state.ws.send(JSON.stringify({ type: 'end_session' }));
    state.ws.close();
  }
  cleanup();
}

function cleanup() {
  state.connected = false;
  state.recording = false;

  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach(t => t.stop());
    state.mediaStream = null;
  }
  if (state.audioCtx) {
    state.audioCtx.close().catch(() => {});
    state.audioCtx = null;
  }

  $('start-btn').disabled = false;
  $('end-btn').disabled = true;
  $('record-btn').disabled = true;
  $('record-btn').classList.remove('recording');
  $('record-icon-mic').classList.remove('hidden');
  $('record-icon-stop').classList.add('hidden');
  $('record-label').textContent = 'Hold to Speak';
  $('text-section').style.display = 'none';

  setMicStatus('Microphone', 'Not connected', false);
  setMicLevel(0);
  hideAction();
  setStatus('Disconnected', '');
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connectWebSocket(url) {
  const ws = new WebSocket(url);
  state.ws = ws;

  ws.onopen = () => {
    state.connected = true;
    setStatus('Connected', 'connected');
    hideAction();
  };

  ws.onmessage = evt => {
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }
    handleServerMessage(msg);
  };

  ws.onerror = () => {
    setStatus('Connection error', 'error');
  };

  ws.onclose = () => {
    state.connected = false;
    setStatus('Disconnected', '');
    hideAction();
  };
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case 'session_started':
      setStatus('Active', 'active');
      addTraceEvent('system', 'SESSION', 'Session started', '');
      break;

    case 'transcription':
      hideAction();
      if (msg.text && msg.text.trim()) {
        addTranscriptEntry('user', msg.text);
        state.turnCount++;
        $('turn-badge').textContent = `Turn ${state.turnCount}`;
      }
      break;

    case 'agent_observation':
    case 'user_transcription':
      showAction('Agent observing...');
      const txt = msg.text || msg.user_message || '';
      if (txt) addTraceEvent('observe', 'OBSERVE', txt, `Iteration ${msg.iteration || state.currentIteration}`);
      break;

    case 'agent_decision': {
      const action = msg.action || '';
      const tool = msg.tool_name ? ` → ${msg.tool_name}` : '';
      const reason = msg.reasoning_summary || '';
      state.currentIteration = msg.iteration || state.currentIteration;
      $('iteration-badge').textContent = `Iteration ${state.currentIteration}`;
      showAction(`Deciding: ${action}${tool}`);
      addTraceEvent('decide', 'DECIDE', `${action}${tool}`, reason, msg.iteration);
      break;
    }

    case 'tool_started':
    case 'tool_requested': {
      const args = JSON.stringify(msg.arguments || {});
      showAction(`Calling ${msg.tool_name}...`);
      addTraceEvent('tool', 'TOOL', msg.tool_name || '', args, msg.iteration);
      break;
    }

    case 'tool_result':
    case 'tool_completed':
    case 'tool_failed': {
      const ok = msg.type !== 'tool_failed';
      const data = JSON.stringify(msg.data || msg.error || {});
      addTraceEvent('result', ok ? 'RESULT' : 'FAILED', msg.tool_name || '', data, msg.iteration);
      break;
    }

    case 'agent_response':
      hideAction();
      if (msg.text) {
        addTranscriptEntry('agent', msg.text);
        addTraceEvent('response', 'SPEAK', msg.text.substring(0, 120) + (msg.text.length > 120 ? '...' : ''), '', msg.iteration);
      }
      break;

    case 'audio':
      if (msg.data) {
        const pcmBytes = base64ToBytes(msg.data);
        audioQueue.enqueue(pcmBytes, msg.sample_rate || 16000);
      }
      break;

    case 'session_ended':
      addTraceEvent('system', 'ENDED', `Reason: ${msg.reason || 'completed'}`, `Turns: ${msg.turns || 0} | Tools: ${msg.tool_calls || 0}`);
      hideAction();
      setStatus('Call ended', '');
      cleanup();
      break;

    case 'error':
      addTraceEvent('error', 'ERROR', msg.message || 'Unknown error', '');
      hideAction();
      break;

    case 'pong':
      break;

    default:
      // Forward unknown events as system trace
      if (msg.type && msg.type !== 'pong') {
        addTraceEvent('system', msg.type.toUpperCase(), JSON.stringify(msg).substring(0, 80), '');
      }
  }
}

// ── Microphone ────────────────────────────────────────────────────────────────

async function initMicrophone() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      }
    });
    state.mediaStream = stream;
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

    // Set up level meter
    const analyser = state.audioCtx.createAnalyser();
    analyser.fftSize = 256;
    const src = state.audioCtx.createMediaStreamSource(stream);
    src.connect(analyser);

    const levelData = new Uint8Array(analyser.frequencyBinCount);
    function updateLevel() {
      if (!state.audioCtx) return;
      analyser.getByteFrequencyData(levelData);
      const avg = levelData.reduce((a, b) => a + b, 0) / levelData.length;
      setMicLevel(avg / 128);
      requestAnimationFrame(updateLevel);
    }
    updateLevel();

    setMicStatus('Microphone', 'Ready — hold button to speak', true);
    return true;
  } catch (err) {
    setMicStatus('Microphone', 'Access denied — use text input', false);
    $('text-section').style.display = 'block';
    return false;
  }
}

// ── Recording ────────────────────────────────────────────────────────────────

async function toggleRecording() {
  if (!state.recording) {
    await startRecording();
  } else {
    stopRecording();
  }
}

async function startRecording() {
  if (!state.mediaStream || !state.connected) return;
  state.recording = true;
  state.audioChunks = [];

  $('record-btn').classList.add('recording');
  $('record-icon-mic').classList.add('hidden');
  $('record-icon-stop').classList.remove('hidden');
  $('record-label').textContent = 'Recording... (click to stop)';
  showAction('Recording your voice...');

  // Use ScriptProcessor to capture raw PCM samples
  const source = state.audioCtx.createMediaStreamSource(state.mediaStream);
  const processor = state.audioCtx.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (e) => {
    if (!state.recording) return;
    const float32 = e.inputBuffer.getChannelData(0);
    const pcm16 = float32ToPCM16(float32);
    state.audioChunks.push(pcm16);

    // Send chunk immediately
    if (state.ws && state.connected) {
      const b64 = bytesToBase64(pcm16);
      state.ws.send(JSON.stringify({
        type: 'audio_chunk',
        data: b64,
        sample_rate: state.audioCtx.sampleRate,
      }));
    }
  };

  source.connect(processor);
  processor.connect(state.audioCtx.destination);
  state.recorder = { source, processor };
}

function stopRecording() {
  if (!state.recording) return;
  state.recording = false;

  if (state.recorder) {
    state.recorder.processor.disconnect();
    state.recorder.source.disconnect();
    state.recorder = null;
  }

  $('record-btn').classList.remove('recording');
  $('record-icon-mic').classList.remove('hidden');
  $('record-icon-stop').classList.add('hidden');
  $('record-label').textContent = 'Hold to Speak';
  showAction('Processing speech...');

  // Signal end of utterance to backend
  if (state.ws && state.connected) {
    state.ws.send(JSON.stringify({ type: 'audio_end' }));
  }
}

// ── Text input ────────────────────────────────────────────────────────────────

function handleTextKey(event) {
  if (event.key === 'Enter') sendText();
}

function sendText() {
  const text = $('text-input').value.trim();
  if (!text || !state.ws || !state.connected) return;
  $('text-input').value = '';

  showAction('Processing...');
  state.ws.send(JSON.stringify({ type: 'text_message', text }));
}

// ── Audio utilities ───────────────────────────────────────────────────────────

function float32ToPCM16(float32Arr) {
  const buf = new ArrayBuffer(float32Arr.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < float32Arr.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Arr[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Uint8Array(buf);
}

async function playPCM16(pcm16Bytes, sampleRate) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
  const n = pcm16Bytes.length / 2;
  const buffer = ctx.createBuffer(1, n, sampleRate);
  const channelData = buffer.getChannelData(0);
  const view = new DataView(pcm16Bytes.buffer, pcm16Bytes.byteOffset, pcm16Bytes.byteLength);
  for (let i = 0; i < n; i++) {
    channelData[i] = view.getInt16(i * 2, true) / 32768.0;
  }
  return new Promise(resolve => {
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    src.onended = () => { ctx.close(); resolve(); };
    src.start();
  });
}

function bytesToBase64(bytes) {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function base64ToBytes(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// ── Transcript UI ────────────────────────────────────────────────────────────

function addTranscriptEntry(role, text) {
  const container = $('transcript');
  const empty = container.querySelector('.transcript-empty');
  if (empty) empty.remove();

  const div = document.createElement('div');
  div.className = `transcript-msg ${role}`;
  div.innerHTML = `
    <span class="msg-role">${role === 'user' ? 'You' : 'Agent'}</span>
    <div class="msg-text">${escapeHtml(text)}</div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function clearTranscript() {
  $('transcript').innerHTML = '<div class="transcript-empty"><p>Start a call to see the conversation transcript here.</p></div>';
  state.turnCount = 0;
  $('turn-badge').textContent = 'Turn 0';
}

// ── Trace UI ──────────────────────────────────────────────────────────────────

function addTraceEvent(type, tag, body, detail, iteration) {
  const list = $('trace-list');
  const empty = list.querySelector('.trace-empty');
  if (empty) empty.remove();

  // Get or create iteration group
  const iterKey = iteration != null ? iteration : 'sys';
  let group = state.iterationGroups[iterKey];
  if (!group && iterKey !== 'sys') {
    group = document.createElement('div');
    group.className = 'trace-iteration';
    const header = document.createElement('div');
    header.className = 'trace-iter-header';
    header.textContent = `Iteration ${iterKey}`;
    group.appendChild(header);
    list.appendChild(group);
    state.iterationGroups[iterKey] = group;
    $('iteration-badge').textContent = `Iteration ${iterKey}`;
  }

  const eventEl = document.createElement('div');
  eventEl.className = `trace-event event-${type}`;
  eventEl.innerHTML = `
    <div class="trace-event-tag">${escapeHtml(tag)}</div>
    <div class="trace-event-body">
      <div>${escapeHtml(body)}</div>
      ${detail ? `<div class="trace-event-detail">${escapeHtml(detail)}</div>` : ''}
    </div>
  `;

  const target = group || list;
  target.appendChild(eventEl);
  list.scrollTop = list.scrollHeight;
}

function clearTrace() {
  $('trace-list').innerHTML = '<div class="trace-empty"><p>Agent reasoning trace will appear here.</p><p class="trace-hint">Each agent iteration shows:<br/>OBSERVE &rarr; DECIDE &rarr; ACT &rarr; RESULT</p></div>';
  state.iterationGroups = {};
  state.currentIteration = 0;
  $('iteration-badge').textContent = 'Iteration 0';
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
