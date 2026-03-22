/**
 * ATP Dashboard API Client
 * Provides API helper functions for communicating with the dashboard backend
 */

const api = {
    /**
     * Make a GET request to the API
     * @param {string} path - API endpoint path (without /api prefix)
     * @returns {Promise<any>} Response data
     */
    async get(path) {
        const token = localStorage.getItem('token');
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        try {
            const res = await fetch(`/api${path}`, { headers });
            if (!res.ok) {
                const errorMessages = {
                    400: 'Invalid request. Please check your input and try again.',
                    401: 'You are not authorized. Please log in and try again.',
                    403: 'Access denied. You do not have permission to access this resource.',
                    404: 'The requested data was not found. It may have been deleted or moved.',
                    408: 'Request timed out. Please check your connection and try again.',
                    429: 'Too many requests. Please wait a moment and try again.',
                    500: 'Server error. Our team has been notified. Please try again later.',
                    502: 'Server is temporarily unavailable. Please try again in a few moments.',
                    503: 'Service is under maintenance. Please try again later.',
                    504: 'Server took too long to respond. Please try again.',
                };
                throw new Error(errorMessages[res.status] || `Something went wrong (Error ${res.status}). Please try again.`);
            }
            return res.json();
        } catch (err) {
            if (err.name === 'TypeError' && err.message.includes('fetch')) {
                throw new Error('Unable to connect to server. Please check your internet connection.');
            }
            throw err;
        }
    },

    /**
     * Make a POST request to the API
     * @param {string} path - API endpoint path (without /api prefix)
     * @param {any} data - Request body data
     * @returns {Promise<any>} Response data
     */
    async post(path, data) {
        const token = localStorage.getItem('token');
        const headers = {
            'Content-Type': 'application/json',
            ...(token && { 'Authorization': `Bearer ${token}` })
        };
        try {
            const res = await fetch(`/api${path}`, {
                method: 'POST',
                headers,
                body: JSON.stringify(data)
            });
            if (!res.ok) {
                const errorMessages = {
                    400: 'Invalid request. Please check your input and try again.',
                    401: 'You are not authorized. Please log in and try again.',
                    403: 'Access denied. You do not have permission to perform this action.',
                    404: 'The requested resource was not found.',
                    422: 'Invalid data format. Please check your input.',
                    429: 'Too many requests. Please wait a moment and try again.',
                    500: 'Server error. Our team has been notified. Please try again later.',
                };
                throw new Error(errorMessages[res.status] || `Something went wrong (Error ${res.status}). Please try again.`);
            }
            return res.json();
        } catch (err) {
            if (err.name === 'TypeError' && err.message.includes('fetch')) {
                throw new Error('Unable to connect to server. Please check your internet connection.');
            }
            throw err;
        }
    },

    /**
     * Make a PUT request to the API
     * @param {string} path - API endpoint path (without /api prefix)
     * @param {any} data - Request body data
     * @returns {Promise<any>} Response data
     */
    async put(path, data) {
        const token = localStorage.getItem('token');
        const headers = {
            'Content-Type': 'application/json',
            ...(token && { 'Authorization': `Bearer ${token}` })
        };
        try {
            const res = await fetch(`/api${path}`, {
                method: 'PUT',
                headers,
                body: JSON.stringify(data)
            });
            if (!res.ok) {
                throw new Error(`Request failed with status ${res.status}`);
            }
            return res.json();
        } catch (err) {
            if (err.name === 'TypeError' && err.message.includes('fetch')) {
                throw new Error('Unable to connect to server. Please check your internet connection.');
            }
            throw err;
        }
    },

    /**
     * Make a DELETE request to the API
     * @param {string} path - API endpoint path (without /api prefix)
     * @returns {Promise<any>} Response data
     */
    async delete(path) {
        const token = localStorage.getItem('token');
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        try {
            const res = await fetch(`/api${path}`, {
                method: 'DELETE',
                headers
            });
            if (!res.ok) {
                throw new Error(`Request failed with status ${res.status}`);
            }
            return res.json();
        } catch (err) {
            if (err.name === 'TypeError' && err.message.includes('fetch')) {
                throw new Error('Unable to connect to server. Please check your internet connection.');
            }
            throw err;
        }
    }
};

// Export for use in other scripts
window.api = api;
