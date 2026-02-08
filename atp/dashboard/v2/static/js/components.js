/**
 * ATP Dashboard React Components
 * Shared React components used across the dashboard
 */

// React hooks shorthand
const { useState, useEffect, useCallback } = React;

// ==================== Event Type Colors ====================

const EVENT_COLORS = {
    tool_call: { bg: 'bg-blue-50', border: 'border-blue-400', text: 'text-blue-700', icon: 'T' },
    llm_request: { bg: 'bg-green-50', border: 'border-green-400', text: 'text-green-700', icon: 'L' },
    reasoning: { bg: 'bg-amber-50', border: 'border-amber-400', text: 'text-amber-700', icon: 'R' },
    error: { bg: 'bg-red-50', border: 'border-red-400', text: 'text-red-700', icon: 'E' },
    progress: { bg: 'bg-purple-50', border: 'border-purple-400', text: 'text-purple-700', icon: 'P' },
};

// ==================== Skeleton Loader Components ====================

/**
 * Basic skeleton element with pulse animation
 */
function SkeletonBox({ className = '', width, height }) {
    const style = {};
    if (width) style.width = width;
    if (height) style.height = height;
    return React.createElement('div', {
        className: `bg-gray-200 rounded skeleton-pulse ${className}`,
        style: style
    });
}

/**
 * Skeleton for text lines
 */
function SkeletonText({ lines = 1, className = '' }) {
    return React.createElement('div', { className: `space-y-2 ${className}` },
        Array.from({ length: lines }).map((_, i) =>
            React.createElement(SkeletonBox, {
                key: i,
                className: 'h-4',
                width: i === lines - 1 && lines > 1 ? '75%' : '100%'
            })
        )
    );
}

/**
 * Skeleton for dashboard summary cards
 */
function SkeletonDashboardSummary() {
    return React.createElement('div', { className: 'grid grid-cols-1 md:grid-cols-4 gap-4 mb-6' },
        [1, 2, 3, 4].map((i) =>
            React.createElement('div', { key: i, className: 'bg-white p-4 rounded-lg shadow' },
                React.createElement(SkeletonBox, { className: 'h-4 w-24 mb-2' }),
                React.createElement(SkeletonBox, { className: 'h-8 w-16' })
            )
        )
    );
}

/**
 * Skeleton for table rows
 */
function SkeletonTableRow({ columns = 5 }) {
    return React.createElement('tr', null,
        Array.from({ length: columns }).map((_, i) =>
            React.createElement('td', { key: i, className: 'px-4 py-3' },
                React.createElement(SkeletonBox, { className: 'h-4', width: i === 0 ? '120px' : '80px' })
            )
        )
    );
}

/**
 * Skeleton for suite list table
 */
function SkeletonSuiteList({ rows = 5 }) {
    return React.createElement('div', { className: 'bg-white rounded-lg shadow overflow-hidden' },
        React.createElement('table', { className: 'min-w-full' },
            React.createElement('thead', { className: 'bg-gray-50' },
                React.createElement('tr', null,
                    ['Suite', 'Agent', 'Status', 'Success Rate', 'Started'].map((header) =>
                        React.createElement('th', {
                            key: header,
                            className: 'px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase'
                        }, header)
                    )
                )
            ),
            React.createElement('tbody', { className: 'divide-y divide-gray-200' },
                Array.from({ length: rows }).map((_, i) =>
                    React.createElement(SkeletonTableRow, { key: i, columns: 5 })
                )
            )
        )
    );
}

/**
 * Skeleton for chart
 */
function SkeletonChart() {
    return React.createElement('div', { className: 'bg-white p-4 rounded shadow' },
        React.createElement(SkeletonBox, { className: 'h-6 w-40 mb-4' }),
        React.createElement('div', { className: 'chart-container flex items-end justify-around px-4' },
            [60, 80, 45, 90, 70, 55, 85].map((height, i) =>
                React.createElement(SkeletonBox, {
                    key: i,
                    className: 'w-8 rounded-t',
                    height: `${height}%`
                })
            )
        )
    );
}

// ==================== Error Boundary Component ====================

/**
 * Error boundary wrapper for catching render errors
 */
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        this.setState({ errorInfo });
        console.error('ErrorBoundary caught an error:', error, errorInfo);
    }

    handleRetry = () => {
        this.setState({ hasError: false, error: null, errorInfo: null });
        if (this.props.onRetry) {
            this.props.onRetry();
        }
    };

    render() {
        if (this.state.hasError) {
            return React.createElement('div', { className: 'bg-red-50 border border-red-200 rounded-lg p-6 text-center' },
                React.createElement('div', { className: 'inline-flex items-center justify-center w-12 h-12 rounded-full bg-red-100 mb-4' },
                    React.createElement('svg', { className: 'w-6 h-6 text-red-600', fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' },
                        React.createElement('path', {
                            strokeLinecap: 'round',
                            strokeLinejoin: 'round',
                            strokeWidth: 2,
                            d: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z'
                        })
                    )
                ),
                React.createElement('h3', { className: 'text-lg font-semibold text-red-800 mb-2' },
                    this.props.title || 'Something went wrong'
                ),
                React.createElement('p', { className: 'text-red-600 mb-4' },
                    this.props.message || 'We encountered an unexpected error. Please try again.'
                ),
                React.createElement('button', {
                    onClick: this.handleRetry,
                    className: 'inline-flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors'
                },
                    React.createElement('svg', { className: 'w-4 h-4', fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' },
                        React.createElement('path', {
                            strokeLinecap: 'round',
                            strokeLinejoin: 'round',
                            strokeWidth: 2,
                            d: 'M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15'
                        })
                    ),
                    'Try Again'
                )
            );
        }

        return this.props.children;
    }
}

/**
 * Functional error display with retry (for async errors)
 */
function ErrorDisplay({ error, onRetry, title = 'Error', message }) {
    const displayMessage = message || error?.message || 'An unexpected error occurred. Please try again.';

    return React.createElement('div', { className: 'bg-red-50 border border-red-200 rounded-lg p-6' },
        React.createElement('div', { className: 'flex items-start gap-4' },
            React.createElement('div', { className: 'flex-shrink-0' },
                React.createElement('div', { className: 'inline-flex items-center justify-center w-10 h-10 rounded-full bg-red-100' },
                    React.createElement('svg', { className: 'w-5 h-5 text-red-600', fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' },
                        React.createElement('path', {
                            strokeLinecap: 'round',
                            strokeLinejoin: 'round',
                            strokeWidth: 2,
                            d: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z'
                        })
                    )
                )
            ),
            React.createElement('div', { className: 'flex-grow' },
                React.createElement('h3', { className: 'text-lg font-semibold text-red-800 mb-1' }, title),
                React.createElement('p', { className: 'text-red-600 mb-3' }, displayMessage),
                onRetry && React.createElement('button', {
                    onClick: onRetry,
                    className: 'inline-flex items-center gap-2 px-4 py-2 bg-red-600 text-white text-sm rounded-md hover:bg-red-700 transition-colors'
                },
                    React.createElement('svg', { className: 'w-4 h-4', fill: 'none', stroke: 'currentColor', viewBox: '0 0 24 24' },
                        React.createElement('path', {
                            strokeLinecap: 'round',
                            strokeLinejoin: 'round',
                            strokeWidth: 2,
                            d: 'M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15'
                        })
                    ),
                    'Try Again'
                )
            )
        )
    );
}

// ==================== Loading Spinner ====================

function LoadingSpinner({ size = 'md', className = '' }) {
    const sizeClasses = {
        sm: 'w-4 h-4',
        md: 'w-6 h-6',
        lg: 'w-8 h-8'
    };
    return React.createElement('div', {
        className: `spinner ${sizeClasses[size]} ${className}`
    });
}

// ==================== Status Badge ====================

function StatusBadge({ status }) {
    const statusClasses = {
        completed: 'bg-green-100 text-green-800',
        failed: 'bg-red-100 text-red-800',
        running: 'bg-yellow-100 text-yellow-800',
        pending: 'bg-gray-100 text-gray-800'
    };
    return React.createElement('span', {
        className: `px-2 py-1 text-xs rounded ${statusClasses[status] || statusClasses.pending}`
    }, status);
}

// ==================== Navigation Component ====================

function Navigation({ currentView, onViewChange, isLoggedIn, onLogout }) {
    const tabs = [
        { id: 'dashboard', label: 'Dashboard' },
        { id: 'suites', label: 'Suites' },
        { id: 'games', label: 'Games' },
        { id: 'compare', label: 'Compare' },
        { id: 'leaderboard', label: 'Leaderboard' },
        { id: 'timeline', label: 'Timeline' },
        { id: 'create', label: 'Create' }
    ];

    return React.createElement('nav', { className: 'bg-white shadow mb-6' },
        React.createElement('div', { className: 'container mx-auto px-4' },
            React.createElement('div', { className: 'flex justify-between items-center h-16' },
                React.createElement('div', { className: 'flex items-center gap-4' },
                    React.createElement('h1', { className: 'text-xl font-bold text-gray-800' }, 'ATP Dashboard'),
                    React.createElement('div', { className: 'flex gap-1' },
                        tabs.map(tab =>
                            React.createElement('button', {
                                key: tab.id,
                                onClick: () => onViewChange(tab.id),
                                className: `px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                                    currentView === tab.id
                                        ? 'bg-blue-100 text-blue-700'
                                        : 'text-gray-600 hover:bg-gray-100'
                                }`
                            }, tab.label)
                        )
                    )
                ),
                React.createElement('div', { className: 'flex items-center gap-4' },
                    isLoggedIn && React.createElement('button', {
                        onClick: onLogout,
                        className: 'text-sm text-gray-600 hover:text-gray-800'
                    }, 'Logout')
                )
            )
        )
    );
}

// Export components globally
window.SkeletonBox = SkeletonBox;
window.SkeletonText = SkeletonText;
window.SkeletonDashboardSummary = SkeletonDashboardSummary;
window.SkeletonTableRow = SkeletonTableRow;
window.SkeletonSuiteList = SkeletonSuiteList;
window.SkeletonChart = SkeletonChart;
window.ErrorBoundary = ErrorBoundary;
window.ErrorDisplay = ErrorDisplay;
window.LoadingSpinner = LoadingSpinner;
window.StatusBadge = StatusBadge;
window.Navigation = Navigation;
window.EVENT_COLORS = EVENT_COLORS;
