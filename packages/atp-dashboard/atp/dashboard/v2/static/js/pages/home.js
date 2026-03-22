/**
 * ATP Dashboard Home Page Scripts
 * Page-specific JavaScript for the home/dashboard view
 */

// This file is loaded by the home.html template
// Additional home page functionality can be added here

/**
 * Initialize the home page
 */
function initHomePage() {
    console.log('Home page initialized');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initHomePage);
} else {
    initHomePage();
}
