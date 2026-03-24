const textInput = document.getElementById('text-input');
const micBtn = document.getElementById('mic-btn');
const translateBtn = document.getElementById('translate-btn');
const glossOutput = document.getElementById('gloss-output');
const videoElement = document.getElementById('sign-video');
const idleVideo = document.getElementById('idle-video');
const videoPlaceholder = document.getElementById('video-placeholder');
const videoLoader = document.getElementById('video-loader');
const splashScreen = document.getElementById('splash-screen');

// Splash Screen Logic
window.addEventListener('load', () => {
    setTimeout(() => {
        splashScreen.classList.add('fade-out');
        if (idleVideo) idleVideo.play().catch(e => console.log("Idle auto-play prevented"));
    }, 3500); // Wait for most of the 4s animation
});

// Speech Recognition Setup
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null; // Initialize later if supported

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';
    recognition.interimResults = false;

    recognition.onstart = () => {
        micBtn.classList.add('recording');
        textInput.placeholder = "Listening...";
    };

    recognition.onend = () => {
        micBtn.classList.remove('recording');
        textInput.placeholder = "Type text or speak...";
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        textInput.value = transcript;
        handleTranslate(); // Auto translate on speech end? Or let user click? Let's auto.
        handleTranslate();
    };
} else {
    micBtn.style.display = 'none';
    console.warn("Speech Recognition not supported in this browser.");
}

// Event Listeners
micBtn.addEventListener('click', () => {
    if (recognition) {
        recognition.start();
    }
});

translateBtn.addEventListener('click', handleTranslate);

// Main Logic
async function handleTranslate() {
    const text = textInput.value.trim();
    if (!text) return;

    setLoading(true);

    try {
        const response = await fetch('/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (response.ok) {
            updateGloss(data.gloss);
            updateVideo(data.video_url);
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }

    } catch (error) {
        console.error('API Error:', error);
        alert('Failed to connect to server.');
    } finally {
        setLoading(false);
    }
}

function updateGloss(glossList) {
    glossOutput.innerHTML = '';
    glossList.forEach(word => {
        const span = document.createElement('span');
        span.className = 'gloss-tag';
        span.textContent = word;
        glossOutput.appendChild(span);
    });
}

function updateVideo(url) {
    if (url) {
        videoPlaceholder.style.display = 'none';
        if (idleVideo) idleVideo.style.display = 'none';

        videoElement.style.display = 'block';
        videoElement.src = url;
        videoElement.load();
        videoElement.play().catch(e => console.log("Auto-play prevented"));
    } else {
        // Handle no video case -> Show placeholder or back to idle?
        if (idleVideo) {
            idleVideo.style.display = 'block';
            idleVideo.play();
            videoElement.style.display = 'none';
        } else {
            videoElement.style.display = 'none';
            videoPlaceholder.style.display = 'flex';
            videoPlaceholder.innerHTML = '<p>No video available for this input.</p>';
        }
    }
}

function setLoading(isLoading) {
    if (isLoading) {
        translateBtn.disabled = true;
        translateBtn.style.opacity = '0.7';
        translateBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';

        videoElement.style.display = 'none';
        if (idleVideo) idleVideo.style.display = 'none';
        videoPlaceholder.style.display = 'none';
        videoLoader.classList.remove('hidden');
    } else {
        translateBtn.disabled = false;
        translateBtn.style.opacity = '1';
        translateBtn.innerHTML = 'Translate to Sign <i class="fa-solid fa-arrow-right"></i>';

        videoLoader.classList.add('hidden');
    }
}
