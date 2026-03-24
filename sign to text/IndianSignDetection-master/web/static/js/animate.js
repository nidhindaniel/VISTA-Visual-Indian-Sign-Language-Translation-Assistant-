/**
 * VISTA - Text/Audio to Sign Language Animation
 * Handles translation API, speech recognition, and video playback.
 */

// ---- DOM Elements ----
const textInput = document.getElementById('animateTextInput');
const animateBtn = document.getElementById('animateBtn');
const clearBtn = document.getElementById('clearBtn');
const micBtn = document.getElementById('micBtn');
const glossOutput = document.getElementById('glossOutput');
const idleVideo = document.getElementById('idleVideo');
const resultVideo = document.getElementById('resultVideo');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const animateLoader = document.getElementById('animateLoader');
const videoActions = document.getElementById('videoActions');
const replayBtn = document.getElementById('replayBtn');
const backToIdleBtn = document.getElementById('backToIdleBtn');

// ---- Speech Recognition ----
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';
    recognition.interimResults = false;

    recognition.onstart = () => {
        micBtn.classList.add('recording');
        textInput.placeholder = 'Listening...';
    };

    recognition.onend = () => {
        micBtn.classList.remove('recording');
        textInput.placeholder = 'Type something like "Hello, how are you?"';
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        textInput.value = transcript;
        handleTranslate();
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        micBtn.classList.remove('recording');
        textInput.placeholder = 'Type something like "Hello, how are you?"';
    };
} else {
    micBtn.style.display = 'none';
    console.warn('Speech Recognition not supported in this browser.');
}

// ---- Event Listeners ----
animateBtn.addEventListener('click', handleTranslate);

clearBtn.addEventListener('click', () => {
    textInput.value = '';
    glossOutput.innerHTML = '<span class="gloss-placeholder">Gloss tokens will appear here after translation...</span>';
    showIdle();
});

micBtn.addEventListener('click', () => {
    if (recognition) {
        try {
            recognition.start();
        } catch (e) {
            // Already started
        }
    }
});

replayBtn.addEventListener('click', () => {
    resultVideo.currentTime = 0;
    resultVideo.play();
});

backToIdleBtn.addEventListener('click', showIdle);

// Enter key to submit
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleTranslate();
    }
});

// ---- Main Translation Logic ----
async function handleTranslate() {
    const text = textInput.value.trim();
    if (!text) return;

    const loader = document.getElementById('vista-global-loader');
    if (loader) loader.classList.remove('hidden');

    setLoading(true);

    try {
        const response = await fetch('/api/animate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (response.ok) {
            updateGloss(data.gloss);
            updateVideo(data.video_url);
        } else {
            showError(data.error || 'Translation failed.');
        }

    } catch (error) {
        console.error('API Error:', error);
        showError('Failed to connect to server.');
    } finally {
        if (loader) setTimeout(() => loader.classList.add('hidden'), 500);
        setLoading(false);
    }
}

// ---- UI Update Functions ----

function updateGloss(glossList) {
    glossOutput.innerHTML = '';
    if (!glossList || glossList.length === 0) {
        glossOutput.innerHTML = '<span class="gloss-placeholder">No glosses generated.</span>';
        return;
    }
    glossList.forEach(word => {
        const tag = document.createElement('span');
        tag.className = 'gloss-tag';
        tag.textContent = word;
        glossOutput.appendChild(tag);
    });
}

function updateVideo(url) {
    if (url) {
        // Hide idle, show result
        if (idleVideo) idleVideo.style.display = 'none';
        videoPlaceholder.style.display = 'none';

        resultVideo.style.display = 'block';
        resultVideo.src = url;
        resultVideo.load();
        resultVideo.play().catch(e => console.log('Auto-play prevented:', e));

        videoActions.style.display = 'flex';
    } else {
        // No video — show placeholder message
        showIdle();
    }
}

function showIdle() {
    resultVideo.style.display = 'none';
    resultVideo.src = '';
    videoPlaceholder.style.display = 'none';
    videoActions.style.display = 'none';

    if (idleVideo) {
        idleVideo.style.display = 'block';
        idleVideo.play().catch(() => { });
    } else {
        videoPlaceholder.style.display = 'flex';
    }
}

function setLoading(isLoading) {
    if (isLoading) {
        animateBtn.disabled = true;
        animateBtn.innerHTML = '<span class="animate-spinner-inline"></span> Processing...';

        if (idleVideo) idleVideo.style.display = 'none';
        resultVideo.style.display = 'none';
        videoPlaceholder.style.display = 'none';
        videoActions.style.display = 'none';
        animateLoader.style.display = 'flex';
    } else {
        animateBtn.disabled = false;
        animateBtn.innerHTML = 'Translate to Sign &rarr;';
        animateLoader.style.display = 'none';
    }
}

function showError(message) {
    glossOutput.innerHTML = `<span class="gloss-error">${message}</span>`;
    showIdle();
}

// ---- Mobile Nav ----
function toggleNav() {
    document.getElementById('navLinks').classList.toggle('open');
}

// ---- Auto-play idle video on load ----
window.addEventListener('load', () => {
    if (idleVideo) {
        idleVideo.play().catch(() => {
            // Auto-play blocked by browser, show placeholder instead
            idleVideo.style.display = 'none';
            videoPlaceholder.style.display = 'flex';
        });
    }
});
