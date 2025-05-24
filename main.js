/**
 * Main functionality for Fruit Quality Classifier
 */

// DOM Elements
const pages = document.querySelectorAll('.page');
const navItems = document.querySelectorAll('.nav-item');
const themeToggle = document.getElementById('themeToggle');
const mobileMenuToggle = document.getElementById('mobileMenuToggle');
const sidebar = document.querySelector('.sidebar');
const methodButtons = document.querySelectorAll('.method-button');
const uploadArea = document.getElementById('uploadArea');
const webcamArea = document.getElementById('webcamArea');
const fileInput = document.getElementById('fileInput');
const webcamVideo = document.getElementById('webcamVideo');
const webcamCanvas = document.getElementById('webcamCanvas');
const captureButton = document.getElementById('captureButton');
const retakeButton = document.getElementById('retakeButton');
const classifyButton = document.getElementById('classifyButton');
const resultsContainer = document.querySelector('.results-container');
const closeResults = document.querySelector('.close-results');
const resultImage = document.getElementById('resultImage');
const heatmapImage = document.getElementById('heatmapImage');
const qualityBadge = document.getElementById('qualityBadge');
const historyList = document.getElementById('historyList');
const historyEmpty = document.querySelector('.history-empty');

// State
let currentPage = 'home';
let currentMethod = 'upload';
let selectedFile = null;
let capturedImage = null;
let stream = null;
let isDarkTheme = false;

// Event Listeners
document.addEventListener('DOMContentLoaded', initApp);
navItems.forEach(item => item.addEventListener('click', changePage));
themeToggle.addEventListener('click', toggleTheme);
mobileMenuToggle.addEventListener('click', toggleMobileMenu);
methodButtons.forEach(button => button.addEventListener('click', changeMethod));
uploadArea.addEventListener('click', triggerFileInput);
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('drop', handleFileDrop);
fileInput.addEventListener('change', handleFileSelect);
captureButton.addEventListener('click', captureWebcam);
retakeButton.addEventListener('click', retakeWebcam);
classifyButton.addEventListener('click', classifyImage);
closeResults.addEventListener('click', closeResultsView);

/**
 * Initialize the application
 */
function initApp() {
    // Check for saved theme preference
    if (localStorage.getItem('theme') === 'dark') {
        enableDarkTheme();
    }
    
    // Show home page by default
    showPage('home');
    
    // Create logo SVG
    createLogo();
}

/**
 * Create SVG logo
 */
function createLogo() {
    const logo = document.getElementById('logo');
    if (!logo) return;
    
    // Create SVG element
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 100 100');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', '#10a37f');
    svg.setAttribute('stroke-width', '4');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    
    // Create apple shape
    const apple = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    apple.setAttribute('d', 'M50 20 C60 10, 80 10, 85 25 C90 40, 80 65, 50 85 C20 65, 10 40, 15 25 C20 10, 40 10, 50 20');
    apple.setAttribute('fill', '#e6f7f2');
    
    // Create stem
    const stem = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    stem.setAttribute('d', 'M50 20 C50 15, 55 5, 60 5');
    
    // Create leaf
    const leaf = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    leaf.setAttribute('d', 'M60 5 C70 5, 75 15, 65 20');
    leaf.setAttribute('fill', '#e6f7f2');
    
    // Add elements to SVG
    svg.appendChild(apple);
    svg.appendChild(stem);
    svg.appendChild(leaf);
    
    // Replace logo placeholder with SVG
    logo.parentNode.replaceChild(svg, logo);
}

/**
 * Change current page
 */
function changePage(event) {
    const pageName = event.currentTarget.dataset.page;
    showPage(pageName);
    
    // Close mobile menu if open
    sidebar.classList.remove('active');
}

/**
 * Show specified page and update navigation
 */
function showPage(pageName) {
    // Update current page
    currentPage = pageName;
    
    // Hide all pages
    pages.forEach(page => page.style.display = 'none');
    
    // Show selected page
    document.getElementById(`${pageName}Page`).style.display = 'block';
    
    // Update navigation
    navItems.forEach(item => item.classList.remove('active'));
    document.querySelector(`[data-page="${pageName}"]`).classList.add('active');
    
    // Special handling for history page
    if (pageName === 'history' && window.auth.isLoggedIn()) {
        loadUserHistory();
    }
}

/**
 * Toggle between light and dark theme
 */
function toggleTheme() {
    if (isDarkTheme) {
        disableDarkTheme();
    } else {
        enableDarkTheme();
    }
}

/**
 * Enable dark theme
 */
function enableDarkTheme() {
    document.documentElement.setAttribute('data-theme', 'dark');
    themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    isDarkTheme = true;
    localStorage.setItem('theme', 'dark');
}

/**
 * Disable dark theme
 */
function disableDarkTheme() {
    document.documentElement.removeAttribute('data-theme');
    themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    isDarkTheme = false;
    localStorage.setItem('theme', 'light');
}

/**
 * Toggle mobile menu
 */
function toggleMobileMenu() {
    sidebar.classList.toggle('active');
}

/**
 * Change upload method (file upload or webcam)
 */
function changeMethod(event) {
    const method = event.currentTarget.dataset.method;
    
    // Update current method
    currentMethod = method;
    
    // Update UI
    methodButtons.forEach(button => button.classList.remove('active'));
    event.currentTarget.classList.add('active');
    
    if (method === 'upload') {
        uploadArea.style.display = 'flex';
        webcamArea.style.display = 'none';
        stopWebcam();
    } else {
        uploadArea.style.display = 'none';
        webcamArea.style.display = 'block';
        startWebcam();
    }
    
    // Reset state
    selectedFile = null;
    capturedImage = null;
    updateClassifyButton();
}

/**
 * Trigger file input click
 */
function triggerFileInput() {
    fileInput.click();
}

/**
 * Handle drag over event
 */
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.add('drag-over');
}

/**
 * Handle file drop event
 */
function handleFileDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.remove('drag-over');
    
    if (event.dataTransfer.files && event.dataTransfer.files[0]) {
        handleFile(event.dataTransfer.files[0]);
    }
}

/**
 * Handle file selection
 */
function handleFileSelect(event) {
    if (event.target.files && event.target.files[0]) {
        handleFile(event.target.files[0]);
    }
}

/**
 * Process selected file
 */
function handleFile(file) {
    // Check if file is an image
    if (!file.type.match('image.*')) {
        alert('Please select an image file');
        return;
    }
    
    // Update state
    selectedFile = file;
    
    // Display selected image in upload area
    const reader = new FileReader();
    reader.onload = function(e) {
        uploadArea.innerHTML = `<img src="${e.target.result}" alt="Selected Image" style="max-width: 100%; max-height: 300px;">`;
    };
    reader.readAsDataURL(file);
    
    // Enable classify button
    updateClassifyButton();
}

/**
 * Start webcam stream
 */
async function startWebcam() {
    try {
        // Get webcam stream
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        
        // Display stream in video element
        webcamVideo.srcObject = stream;
        webcamVideo.style.display = 'block';
        webcamCanvas.style.display = 'none';
        captureButton.style.display = 'block';
        retakeButton.style.display = 'none';
        
        // Reset state
        capturedImage = null;
        updateClassifyButton();
    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Could not access webcam. Please allow camera access or use file upload instead.');
        
        // Switch back to upload method
        document.querySelector('[data-method="upload"]').click();
    }
}

/**
 * Stop webcam stream
 */
function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

/**
 * Capture image from webcam
 */
function captureWebcam() {
    if (!webcamVideo.srcObject) return;
    
    // Get video dimensions
    const videoWidth = webcamVideo.videoWidth;
    const videoHeight = webcamVideo.videoHeight;
    
    // Set canvas dimensions to match video
    webcamCanvas.width = videoWidth;
    webcamCanvas.height = videoHeight;
    
    // Draw video frame to canvas
    const ctx = webcamCanvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0, videoWidth, videoHeight);
    
    // Get image data
    capturedImage = webcamCanvas.toDataURL('image/jpeg');
    
    // Update UI
    webcamVideo.style.display = 'none';
    webcamCanvas.style.display = 'block';
    captureButton.style.display = 'none';
    retakeButton.style.display = 'block';
    
    // Enable classify button
    updateClassifyButton();
}

/**
 * Retake webcam image
 */
function retakeWebcam() {
    // Reset state
    capturedImage = null;
    
    // Update UI
    webcamVideo.style.display = 'block';
    webcamCanvas.style.display = 'none';
    captureButton.style.display = 'block';
    retakeButton.style.display = 'none';
    
    // Disable classify button
    updateClassifyButton();
}

/**
 * Update classify button state
 */
function updateClassifyButton() {
    if ((currentMethod === 'upload' && selectedFile) || 
        (currentMethod === 'webcam' && capturedImage)) {
        classifyButton.disabled = false;
    } else {
        classifyButton.disabled = true;
    }
}

/**
 * Classify the selected or captured image
 */
async function classifyImage() {
    // Check if user is logged in
    if (!window.auth.isLoggedIn()) {
        // Show login modal
        document.getElementById('authButton').click();
        return;
    }
    
    try {
        // Show loading
        window.auth.showLoading();
        
        let formData;
        
        if (currentMethod === 'upload' && selectedFile) {
            // Create form data with file
            formData = new FormData();
            formData.append('file', selectedFile);
            
            // Send prediction request
            const response = await fetch('/api/prediction/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Prediction failed');
            }
            
            const result = await response.json();
            displayResults(result);
        } else if (currentMethod === 'webcam' && capturedImage) {
            // Send prediction request with base64 image
            const response = await fetch('/api/prediction/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image_data: capturedImage
                })
            });
            
            if (!response.ok) {
                throw new Error('Prediction failed');
            }
            
            const result = await response.json();
            displayResults(result);
        }
    } catch (error) {
        console.error('Classification error:', error);
        alert('An error occurred during classification. Please try again.');
    } finally {
        window.auth.hideLoading();
    }
}

/**
 * Display classification results
 */
function displayResults(result) {
    // Set result image
    resultImage.src = result.image_url;
    
    // Set heatmap image if available
    if (result.heatmap_url) {
        heatmapImage.src = result.heatmap_url;
        document.querySelector('.heatmap-image').style.display = 'block';
    } else {
        document.querySelector('.heatmap-image').style.display = 'none';
    }
    
    // Set quality badge
    qualityBadge.className = 'quality-badge ' + result.predicted_class;
    qualityBadge.innerHTML = `
        <span class="quality-label">${capitalizeFirstLetter(result.predicted_class)}</span>
        <span class="quality-confidence">${Math.round(result.confidence * 100)}%</span>
    `;
    
    // Set confidence bars
    const confidenceBars = document.querySelector('.confidence-bars');
    confidenceBars.innerHTML = '';
    
    for (const [className, confidence] of Object.entries(result.confidences)) {
        const confidencePercent = Math.round(confidence * 100);
        
        confidenceBars.innerHTML += `
            <div class="confidence-item">
                <span class="label">${capitalizeFirstLetter(className)}</span>
                <div class="bar-container">
                    <div class="bar ${className}" style="width: ${confidencePercent}%;"></div>
                    <span class="value">${confidencePercent}%</span>
                </div>
            </div>
        `;
    }
    
    // Set explanation
    const explanation = document.querySelector('.result-explanation');
    explanation.innerHTML = `
        <h4>What does this mean?</h4>
        <p>This fruit appears to be of <strong>${result.predicted_class} quality</strong>. 
        ${getExplanationText(result.predicted_class, Math.round(result.confidence * 100))}</p>
    `;
    
    // Show results container
    resultsContainer.style.display = 'block';
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Get explanation text based on quality class and confidence
 */
function getExplanationText(qualityClass, confidence) {
    if (qualityClass === 'good') {
        if (confidence > 90) {
            return 'The fruit has excellent appearance with no visible defects. The model is highly confident in this classification.';
        } else {
            return 'The fruit appears to be of good quality with minimal defects, though there may be some minor imperfections.';
        }
    } else if (qualityClass === 'average') {
        if (confidence > 90) {
            return 'The fruit has some visible minor defects or slight discoloration. The model is highly confident in this classification.';
        } else {
            return 'The fruit shows some signs of imperfections, though it may still be suitable for consumption.';
        }
    } else {
        if (confidence > 90) {
            return 'The fruit has significant defects, bruising, or spoilage. The model is highly confident in this classification.';
        } else {
            return 'The fruit shows signs of defects or spoilage that may affect its quality and taste.';
        }
    }
}

/**
 * Close results view
 */
function closeResultsView() {
    resultsContainer.style.display = 'none';
}

/**
 * Load user prediction history
 */
async function loadUserHistory() {
    if (!window.auth.isLoggedIn()) return;
    
    try {
        const response = await fetch('/api/auth/history');
        
        if (!response.ok) {
            throw new Error('Failed to load history');
        }
        
        const data = await response.json();
        
        if (data.history && data.history.length > 0) {
            // Show history
            renderHistory(data.history);
            historyEmpty.style.display = 'none';
     
(Content truncated due to size limit. Use line ranges to read in chunks)