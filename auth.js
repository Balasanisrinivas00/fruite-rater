/**
 * Authentication functionality for Fruit Quality Classifier
 */

// DOM Elements
const authButton = document.getElementById('authButton');
const authModal = document.getElementById('authModal');
const closeModal = document.querySelector('.close-modal');
const authTabs = document.querySelectorAll('.auth-tab');
const loginForm = document.getElementById('loginForm');
const registerForm = document.getElementById('registerForm');
const loginSubmit = document.getElementById('loginSubmit');
const registerSubmit = document.getElementById('registerSubmit');
const loginMessage = document.getElementById('loginMessage');
const registerMessage = document.getElementById('registerMessage');
const usernameDisplay = document.querySelector('.username');

// State
let isLoggedIn = false;
let currentUser = null;

// Event Listeners
authButton.addEventListener('click', toggleAuthModal);
closeModal.addEventListener('click', closeAuthModal);
authTabs.forEach(tab => tab.addEventListener('click', switchAuthTab));
loginSubmit.addEventListener('click', handleLogin);
registerSubmit.addEventListener('click', handleRegister);

// Check if user is already logged in
checkAuthStatus();

/**
 * Toggle authentication modal visibility
 */
function toggleAuthModal() {
    if (isLoggedIn) {
        // If logged in, log out instead of showing modal
        logout();
    } else {
        // Show login modal
        authModal.classList.add('active');
    }
}

/**
 * Close authentication modal
 */
function closeAuthModal() {
    authModal.classList.remove('active');
    clearAuthForms();
}

/**
 * Switch between login and register tabs
 */
function switchAuthTab(event) {
    const tabName = event.target.dataset.tab;
    
    // Update active tab
    authTabs.forEach(tab => tab.classList.remove('active'));
    event.target.classList.add('active');
    
    // Show corresponding form
    if (tabName === 'login') {
        loginForm.style.display = 'block';
        registerForm.style.display = 'none';
    } else {
        loginForm.style.display = 'none';
        registerForm.style.display = 'block';
    }
    
    // Clear messages
    loginMessage.textContent = '';
    registerMessage.textContent = '';
}

/**
 * Handle login form submission
 */
async function handleLogin() {
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value;
    
    // Validate inputs
    if (!username || !password) {
        loginMessage.textContent = 'Please enter both username and password';
        return;
    }
    
    try {
        // Show loading
        showLoading();
        
        // Send login request
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Login successful
            loginMessage.textContent = 'Login successful!';
            loginMessage.classList.add('success');
            
            // Update UI for logged in state
            updateLoggedInState(username);
            
            // Close modal after short delay
            setTimeout(() => {
                closeAuthModal();
            }, 1000);
            
            // Load user history
            loadUserHistory();
        } else {
            // Login failed
            loginMessage.textContent = data.error || 'Login failed';
            loginMessage.classList.remove('success');
        }
    } catch (error) {
        loginMessage.textContent = 'An error occurred. Please try again.';
        console.error('Login error:', error);
    } finally {
        hideLoading();
    }
}

/**
 * Handle register form submission
 */
async function handleRegister() {
    const username = document.getElementById('registerUsername').value.trim();
    const password = document.getElementById('registerPassword').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    
    // Validate inputs
    if (!username || !password) {
        registerMessage.textContent = 'Please enter both username and password';
        return;
    }
    
    if (password !== confirmPassword) {
        registerMessage.textContent = 'Passwords do not match';
        return;
    }
    
    try {
        // Show loading
        showLoading();
        
        // Send register request
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Registration successful
            registerMessage.textContent = 'Registration successful! You can now log in.';
            registerMessage.classList.add('success');
            
            // Switch to login tab after short delay
            setTimeout(() => {
                authTabs[0].click();
            }, 1500);
        } else {
            // Registration failed
            registerMessage.textContent = data.error || 'Registration failed';
            registerMessage.classList.remove('success');
        }
    } catch (error) {
        registerMessage.textContent = 'An error occurred. Please try again.';
        console.error('Registration error:', error);
    } finally {
        hideLoading();
    }
}

/**
 * Check if user is already logged in
 */
async function checkAuthStatus() {
    try {
        const response = await fetch('/api/auth/user');
        
        if (response.ok) {
            const data = await response.json();
            updateLoggedInState(data.username);
            loadUserHistory();
        }
    } catch (error) {
        console.error('Auth status check error:', error);
    }
}

/**
 * Log out the current user
 */
async function logout() {
    try {
        // Show loading
        showLoading();
        
        // Send logout request
        const response = await fetch('/api/auth/logout', {
            method: 'POST'
        });
        
        if (response.ok) {
            // Update UI for logged out state
            updateLoggedOutState();
            
            // Clear history
            clearHistory();
        }
    } catch (error) {
        console.error('Logout error:', error);
    } finally {
        hideLoading();
    }
}

/**
 * Update UI for logged in state
 */
function updateLoggedInState(username) {
    isLoggedIn = true;
    currentUser = username;
    
    // Update button text
    authButton.textContent = 'Logout';
    
    // Update username display
    usernameDisplay.textContent = username;
    
    // Enable history tab
    document.querySelector('[data-page="history"]').classList.remove('disabled');
}

/**
 * Update UI for logged out state
 */
function updateLoggedOutState() {
    isLoggedIn = false;
    currentUser = null;
    
    // Update button text
    authButton.textContent = 'Login';
    
    // Update username display
    usernameDisplay.textContent = 'Guest';
    
    // If on history page, switch to home
    if (document.getElementById('historyPage').style.display !== 'none') {
        document.querySelector('[data-page="home"]').click();
    }
}

/**
 * Clear authentication forms
 */
function clearAuthForms() {
    document.getElementById('loginUsername').value = '';
    document.getElementById('loginPassword').value = '';
    document.getElementById('registerUsername').value = '';
    document.getElementById('registerPassword').value = '';
    document.getElementById('confirmPassword').value = '';
    loginMessage.textContent = '';
    registerMessage.textContent = '';
}

/**
 * Show loading overlay
 */
function showLoading() {
    document.getElementById('loadingOverlay').classList.add('active');
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

// Export functions for use in main.js
window.auth = {
    isLoggedIn: () => isLoggedIn,
    getCurrentUser: () => currentUser,
    showLoading,
    hideLoading
};
