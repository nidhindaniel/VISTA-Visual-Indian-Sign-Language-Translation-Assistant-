/**
 * VISTA - Real-time Sign Language Translation
 * WebSocket client for webcam → server prediction
 */

// ---- State ----
let socket = null;
let stream = null;
let captureInterval = null;
let isRunning = false;
let sentence = '';
let lastSign = '';
let sameSignCount = 0;
const HOLD_THRESHOLD = 8;  // frames with same sign before adding to sentence
const CAPTURE_FPS = 10;    // frames per second sent to server

// ---- DOM ----
const video = document.getElementById('webcamVideo');
const canvas = document.getElementById('overlayCanvas');
const ctx = canvas.getContext('2d');
const webcamStatus = document.getElementById('webcamStatus');
const statusText = document.getElementById('statusText');
const currentSign = document.getElementById('currentSign');
const confidenceSection = document.getElementById('confidenceSection');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const sentenceOutput = document.getElementById('sentenceOutput');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

// ---- Socket Connection ----
function connectSocket() {
    socket = io({ transports: ['websocket', 'polling'] });

    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('prediction', (data) => {
        updatePrediction(data);
    });

    socket.on('disconnect', () => {
        console.log('Disconnected');
    });
}

// ---- Camera ----
async function startCamera() {
    const loader = document.getElementById('vista-global-loader');
    if (loader) loader.classList.remove('hidden');

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });

        video.srcObject = stream;
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            if (loader) setTimeout(() => loader.classList.add('hidden'), 500);
        };

        isRunning = true;
        webcamStatus.classList.add('active');
        statusText.textContent = 'Live';
        startBtn.disabled = true;
        stopBtn.disabled = false;

        if (!socket) connectSocket();

        // Start sending frames
        captureInterval = setInterval(captureAndSend, 1000 / CAPTURE_FPS);

    } catch (err) {
        if (loader) loader.classList.add('hidden');
        console.error('Camera error:', err);
        statusText.textContent = 'Camera access denied';
        
        let errorMsg = 'Camera access denied. Please allow camera permissions.';
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            errorMsg = 'Camera blocked due to insecure context. If accessing from another device, the server must use HTTPS. If on the same device, use http://localhost:8080 instead of an IP address.';
        } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            errorMsg = 'Camera access was blocked by the browser. Please click the lock/camera icon in the address bar to allow access and reload.';
        } else if (err.name === 'NotFoundError') {
            errorMsg = 'No camera device found on your system.';
        }

        currentSign.innerHTML = `<div class="no-hand" style="font-size: 1rem; padding: 20px; line-height: 1.5;">${errorMsg}</div>`;
    }
}

function stopCamera() {
    isRunning = false;

    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    webcamStatus.classList.remove('active');
    statusText.textContent = 'Stopped';
    startBtn.disabled = false;
    stopBtn.disabled = true;
    currentSign.innerHTML = '<div class="no-hand">Camera stopped</div>';
    confidenceSection.style.display = 'none';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ---- Frame Capture & Send ----
function captureAndSend() {
    if (!isRunning || !socket || !video.videoWidth) return;

    // Create temp canvas for capturing frame
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');

    // Mirror the frame to match what the user sees
    tempCtx.translate(tempCanvas.width, 0);
    tempCtx.scale(-1, 1);
    tempCtx.drawImage(video, 0, 0);

    // Send as base64 JPEG
    const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.7);
    socket.emit('frame', { image: dataUrl });
}

// ---- Update Prediction ----
function updatePrediction(data) {
    // Clear overlay
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Mirror the canvas drawing
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    if (data.hand_detected && data.landmarks) {
        drawHandSkeleton(data.landmarks);
    }

    ctx.restore();

    if (data.hand_detected && data.label) {
        const conf = Math.round(data.confidence * 100);
        currentSign.innerHTML = `<div class="letter">${data.label}</div>`;
        confidenceSection.style.display = 'block';
        confidenceValue.textContent = `${conf}%`;
        confidenceFill.style.width = `${conf}%`;

        // Color based on confidence
        if (conf >= 80) {
            confidenceFill.style.background = 'linear-gradient(135deg, #00b894, #55efc4)';
        } else if (conf >= 50) {
            confidenceFill.style.background = 'linear-gradient(135deg, #fdcb6e, #ffeaa7)';
        } else {
            confidenceFill.style.background = 'linear-gradient(135deg, #e17055, #fab1a0)';
        }

        // Sentence builder: add character after HOLD_THRESHOLD consistent frames
        if (data.label === lastSign && conf >= 60) {
            sameSignCount++;
            if (sameSignCount === HOLD_THRESHOLD) {
                sentence += data.label;
                updateSentenceDisplay();
            }
        } else {
            lastSign = data.label;
            sameSignCount = 0;
        }

    } else {
        currentSign.innerHTML = '<div class="no-hand">Show your hand to the camera</div>';
        confidenceSection.style.display = 'none';
        lastSign = '';
        sameSignCount = 0;
    }
}

// ---- Draw Hand Skeleton ----
function drawHandSkeleton(landmarks) {
    if (!landmarks || landmarks.length === 0) return;

    const w = canvas.width;
    const h = canvas.height;

    // MediaPipe hand connections
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
        [0, 5], [5, 6], [6, 7], [7, 8],       // index
        [0, 9], [9, 10], [10, 11], [11, 12],  // middle (via 0 through 5-9)
        [0, 13], [13, 14], [14, 15], [15, 16],// ring
        [0, 17], [17, 18], [18, 19], [19, 20],// pinky
        [5, 9], [9, 13], [13, 17]           // palm
    ];

    // Draw connections
    ctx.strokeStyle = 'rgba(85, 239, 196, 0.8)';
    ctx.lineWidth = 2;

    connections.forEach(([a, b]) => {
        if (landmarks[a] && landmarks[b]) {
            ctx.beginPath();
            ctx.moveTo(landmarks[a][0] * w, landmarks[a][1] * h);
            ctx.lineTo(landmarks[b][0] * w, landmarks[b][1] * h);
            ctx.stroke();
        }
    });

    // Draw points
    landmarks.forEach((lm, i) => {
        if (!lm) return;
        const x = lm[0] * w;
        const y = lm[1] * h;

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = i === 0 ? '#6c5ce7' : '#55efc4';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.6)';
        ctx.lineWidth = 1;
        ctx.stroke();
    });
}

// ---- Sentence Controls ----
function updateSentenceDisplay() {
    sentenceOutput.textContent = sentence || '\u00A0';
    sentenceOutput.style.color = '';
    sentenceOutput.style.fontSize = '';
}

function addSpace() {
    sentence += ' ';
    updateSentenceDisplay();
}

function backspace() {
    sentence = sentence.slice(0, -1);
    updateSentenceDisplay();
}

function clearSentence() {
    sentence = '';
    sentenceOutput.innerHTML = '<span style="color: var(--text-muted); font-size: 0.9rem;">Recognized signs will appear here...</span>';
}

function copySentence() {
    if (sentence) {
        navigator.clipboard.writeText(sentence).then(() => {
            const btn = event.target;
            btn.textContent = 'Copied!';
            setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
        });
    }
}

// Mobile nav
function toggleNav() {
    document.getElementById('navLinks').classList.toggle('open');
}
