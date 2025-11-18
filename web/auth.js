/**
 * Authentication module for Zeus Splunk Query Assistant
 */

const AUTH_TOKEN_KEY = 'zeus_auth_token';
const USER_DATA_KEY = 'zeus_user_data';

// Authentication state
let currentUser = null;
let authToken = null;

// Initialize authentication on page load
function initAuth() {
    // Load token and user data from localStorage
    authToken = localStorage.getItem(AUTH_TOKEN_KEY);
    const userData = localStorage.getItem(USER_DATA_KEY);

    if (authToken && userData) {
        try {
            currentUser = JSON.parse(userData);
            updateUIForAuthenticatedUser();
        } catch (e) {
            console.error('Failed to parse user data:', e);
            logout();
        }
    } else {
        showLoginModal();
    }
}

// Show login modal
function showLoginModal() {
    document.getElementById('auth-modal').style.display = 'flex';
    document.getElementById('login-form-container').style.display = 'block';
    document.getElementById('register-form-container').style.display = 'none';
    document.getElementById('main-app').style.display = 'none';
}

// Show register modal
function showRegisterModal() {
    document.getElementById('auth-modal').style.display = 'flex';
    document.getElementById('login-form-container').style.display = 'none';
    document.getElementById('register-form-container').style.display = 'block';
}

// Switch to login form
function switchToLogin() {
    document.getElementById('login-form-container').style.display = 'block';
    document.getElementById('register-form-container').style.display = 'none';
}

// Switch to register form
function switchToRegister() {
    document.getElementById('login-form-container').style.display = 'none';
    document.getElementById('register-form-container').style.display = 'block';
}

// Handle login
async function handleLogin(event) {
    event.preventDefault();

    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    const errorEl = document.getElementById('login-error');
    const submitBtn = event.target.querySelector('button[type="submit"]');

    errorEl.textContent = '';
    submitBtn.disabled = true;
    submitBtn.textContent = 'Logging in...';

    try {
        const response = await fetch(`${API_URL}/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Login failed');
        }

        const data = await response.json();

        // Store token and user data
        authToken = data.access_token;
        currentUser = data.user;
        localStorage.setItem(AUTH_TOKEN_KEY, authToken);
        localStorage.setItem(USER_DATA_KEY, JSON.stringify(currentUser));

        // Hide modal and show main app
        document.getElementById('auth-modal').style.display = 'none';
        document.getElementById('main-app').style.display = 'flex';
        updateUIForAuthenticatedUser();

        // Reset form
        event.target.reset();

    } catch (error) {
        errorEl.textContent = error.message;
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Login';
    }
}

// Handle registration
async function handleRegister(event) {
    event.preventDefault();

    const username = document.getElementById('register-username').value;
    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;
    const fullName = document.getElementById('register-fullname').value;
    const errorEl = document.getElementById('register-error');
    const submitBtn = event.target.querySelector('button[type="submit"]');

    errorEl.textContent = '';
    submitBtn.disabled = true;
    submitBtn.textContent = 'Creating account...';

    try {
        const response = await fetch(`${API_URL}/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username,
                email,
                password,
                full_name: fullName || null,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Registration failed');
        }

        const data = await response.json();

        // Store token and user data
        authToken = data.access_token;
        currentUser = data.user;
        localStorage.setItem(AUTH_TOKEN_KEY, authToken);
        localStorage.setItem(USER_DATA_KEY, JSON.stringify(currentUser));

        // Hide modal and show main app
        document.getElementById('auth-modal').style.display = 'none';
        document.getElementById('main-app').style.display = 'flex';
        updateUIForAuthenticatedUser();

        // Reset form
        event.target.reset();

    } catch (error) {
        errorEl.textContent = error.message;
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Create Account';
    }
}

// Logout
function logout() {
    authToken = null;
    currentUser = null;
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(USER_DATA_KEY);

    // Hide user info header
    const userInfoEl = document.getElementById('user-info');
    if (userInfoEl) {
        userInfoEl.style.display = 'none';
    }

    showLoginModal();
}

// Update UI for authenticated user
function updateUIForAuthenticatedUser() {
    if (currentUser) {
        const userInfoEl = document.getElementById('user-info');
        if (userInfoEl) {
            userInfoEl.innerHTML = `
                <span class="user-name">${escapeHtml(currentUser.username)}</span>
                ${currentUser.is_admin ? '<span class="admin-badge">Admin</span>' : ''}
                <button onclick="logout()" class="logout-btn">Logout</button>
            `;
            userInfoEl.style.display = 'flex';
        }
    }
}

// Get auth headers for API requests
function getAuthHeaders() {
    const headers = {
        'Content-Type': 'application/json',
    };

    if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
    }

    return headers;
}

// Check if user is authenticated
function isAuthenticated() {
    return authToken !== null && currentUser !== null;
}

// Helper function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
