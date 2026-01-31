"""FastAPI application for ATP Dashboard."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from atp.dashboard.api import router as api_router
from atp.dashboard.database import init_database


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover
    """Application lifespan handler."""
    # Initialize database on startup
    db_url = os.getenv("ATP_DATABASE_URL")
    echo = os.getenv("ATP_DEBUG", "false").lower() == "true"
    await init_database(url=db_url, echo=echo)
    yield


# Create FastAPI application
app = FastAPI(
    title="ATP Dashboard",
    description="Web dashboard for Agent Test Platform results",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ATP_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix="/api")

# Static files directory for React frontend
STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


def create_index_html() -> str:
    """Create a basic index.html for development/demo purposes."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATP Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .chart-container { position: relative; height: 300px; }
        .matrix-scroll-container { max-width: 100%; overflow-x: auto; }
        .matrix-scroll-container::-webkit-scrollbar { height: 8px; }
        .matrix-scroll-container::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 4px; }
        .matrix-scroll-container::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 4px; }
        .matrix-scroll-container::-webkit-scrollbar-thumb:hover { background: #a1a1a1; }
        /* Timeline styles */
        .timeline-container { position: relative; overflow-x: auto; }
        .timeline-track { position: relative; min-height: 40px; }
        .timeline-marker { transition: all 0.15s ease; }
        .timeline-marker:hover { transform: scale(1.1); z-index: 10; }
        /* Skeleton loader animation */
        @keyframes skeleton-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .skeleton-pulse { animation: skeleton-pulse 1.5s ease-in-out infinite; }
        /* Focus styles for keyboard navigation */
        .timeline-event-focusable:focus {
            outline: 2px solid #3B82F6;
            outline-offset: 2px;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
        }
        .timeline-event-focusable:focus:not(:focus-visible) {
            outline: none;
            box-shadow: none;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div id="root"></div>
    <script type="text/babel">
        const { useState, useEffect, useCallback } = React;

        // API helper with user-friendly error messages
        const api = {
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
            }
        };

        // ==================== Skeleton Loader Components ====================

        // Basic skeleton element with pulse animation
        function SkeletonBox({ className = '', width, height }) {
            const style = {};
            if (width) style.width = width;
            if (height) style.height = height;
            return (
                <div
                    className={`bg-gray-200 rounded skeleton-pulse ${className}`}
                    style={style}
                />
            );
        }

        // Skeleton for text lines
        function SkeletonText({ lines = 1, className = '' }) {
            return (
                <div className={`space-y-2 ${className}`}>
                    {Array.from({ length: lines }).map((_, i) => (
                        <SkeletonBox
                            key={i}
                            className="h-4"
                            width={i === lines - 1 && lines > 1 ? '75%' : '100%'}
                        />
                    ))}
                </div>
            );
        }

        // Skeleton for dashboard summary cards
        function SkeletonDashboardSummary() {
            return (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                    {[1, 2, 3, 4].map((i) => (
                        <div key={i} className="bg-white p-4 rounded-lg shadow">
                            <SkeletonBox className="h-4 w-24 mb-2" />
                            <SkeletonBox className="h-8 w-16" />
                        </div>
                    ))}
                </div>
            );
        }

        // Skeleton for table rows
        function SkeletonTableRow({ columns = 5 }) {
            return (
                <tr>
                    {Array.from({ length: columns }).map((_, i) => (
                        <td key={i} className="px-4 py-3">
                            <SkeletonBox className="h-4" width={i === 0 ? '120px' : '80px'} />
                        </td>
                    ))}
                </tr>
            );
        }

        // Skeleton for suite list table
        function SkeletonSuiteList({ rows = 5 }) {
            return (
                <div className="bg-white rounded-lg shadow overflow-hidden">
                    <table className="min-w-full">
                        <thead className="bg-gray-50">
                            <tr>
                                {['Suite', 'Agent', 'Status', 'Success Rate', 'Started'].map((header) => (
                                    <th key={header} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                                        {header}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            {Array.from({ length: rows }).map((_, i) => (
                                <SkeletonTableRow key={i} columns={5} />
                            ))}
                        </tbody>
                    </table>
                </div>
            );
        }

        // Skeleton for metrics panel
        function SkeletonMetricsPanel({ agents = 2 }) {
            return (
                <div className="bg-white rounded-lg shadow">
                    <div className="p-4 border-b bg-gray-50 rounded-t-lg">
                        <SkeletonBox className="h-6 w-48 mb-2" />
                        <SkeletonBox className="h-4 w-32" />
                    </div>
                    <div className={`grid gap-4 p-4 ${
                        agents === 1 ? 'grid-cols-1' :
                        agents === 2 ? 'grid-cols-1 md:grid-cols-2' :
                        'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'
                    }`}>
                        {Array.from({ length: agents }).map((_, i) => (
                            <div key={i} className="border rounded-lg overflow-hidden">
                                <div className="p-3 bg-gray-50 border-b">
                                    <SkeletonBox className="h-5 w-24" />
                                </div>
                                <div className="grid grid-cols-2 gap-2 p-3">
                                    {[1, 2, 3, 4, 5, 6].map((j) => (
                                        <div key={j} className="p-2 rounded bg-gray-50">
                                            <SkeletonBox className="h-3 w-12 mb-2" />
                                            <SkeletonBox className="h-5 w-16" />
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            );
        }

        // Skeleton for leaderboard matrix
        function SkeletonLeaderboardMatrix({ rows = 5, cols = 3 }) {
            return (
                <div className="bg-white rounded-lg shadow">
                    <div className="p-4 border-b flex items-center justify-between">
                        <div>
                            <SkeletonBox className="h-6 w-48 mb-2" />
                            <SkeletonBox className="h-4 w-32" />
                        </div>
                        <div className="flex gap-2">
                            <SkeletonBox className="h-8 w-24 rounded" />
                            <SkeletonBox className="h-8 w-24 rounded" />
                        </div>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="min-w-full">
                            <thead>
                                <tr>
                                    <th className="px-3 py-3 bg-gray-100">
                                        <SkeletonBox className="h-4 w-16" />
                                    </th>
                                    <th className="px-3 py-3 bg-gray-100">
                                        <SkeletonBox className="h-4 w-12" />
                                    </th>
                                    {Array.from({ length: cols }).map((_, i) => (
                                        <th key={i} className="px-3 py-3 bg-gray-100 min-w-[120px]">
                                            <div className="flex flex-col items-center gap-1">
                                                <SkeletonBox className="h-5 w-20" />
                                                <SkeletonBox className="h-3 w-16" />
                                            </div>
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {Array.from({ length: rows }).map((_, i) => (
                                    <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                                        <td className="px-3 py-2">
                                            <SkeletonBox className="h-4 w-32 mb-1" />
                                            <SkeletonBox className="h-3 w-16" />
                                        </td>
                                        <td className="px-3 py-2 text-center">
                                            <SkeletonBox className="h-4 w-10 mx-auto" />
                                        </td>
                                        {Array.from({ length: cols }).map((_, j) => (
                                            <td key={j} className="px-3 py-2 text-center">
                                                <SkeletonBox className="h-5 w-10 mx-auto mb-1" />
                                                <SkeletonBox className="h-3 w-8 mx-auto" />
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            );
        }

        // Skeleton for timeline
        function SkeletonTimeline({ rows = 2 }) {
            return (
                <div className="bg-white rounded-lg shadow p-4">
                    {/* Time scale skeleton */}
                    <div className="h-8 border-b border-gray-300 bg-gray-50 mb-4 flex items-end justify-between px-4">
                        {[0, 1, 2, 3, 4].map((i) => (
                            <SkeletonBox key={i} className="h-4 w-10" />
                        ))}
                    </div>
                    {/* Timeline rows skeleton */}
                    {Array.from({ length: rows }).map((_, i) => (
                        <div key={i} className="mb-4">
                            <div className="flex items-center justify-between mb-2 px-2">
                                <SkeletonBox className="h-5 w-32" />
                                <SkeletonBox className="h-4 w-20" />
                            </div>
                            <div className="h-10 bg-gray-100 rounded border border-gray-200 relative overflow-hidden">
                                {/* Random event markers */}
                                {[15, 35, 50, 70, 85].map((pos, j) => (
                                    <SkeletonBox
                                        key={j}
                                        className="absolute top-1/2 -translate-y-1/2 h-6 w-8 rounded"
                                        width="2%"
                                        style={{ left: `${pos}%` }}
                                    />
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            );
        }

        // Skeleton for chart
        function SkeletonChart() {
            return (
                <div className="bg-white p-4 rounded shadow">
                    <SkeletonBox className="h-6 w-40 mb-4" />
                    <div className="chart-container flex items-end justify-around px-4">
                        {[60, 80, 45, 90, 70, 55, 85].map((height, i) => (
                            <SkeletonBox
                                key={i}
                                className="w-8 rounded-t"
                                height={`${height}%`}
                            />
                        ))}
                    </div>
                </div>
            );
        }

        // ==================== Error Boundary Component ====================

        // Error boundary wrapper for catching render errors
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
                    return (
                        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
                            <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-red-100 mb-4">
                                <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                </svg>
                            </div>
                            <h3 className="text-lg font-semibold text-red-800 mb-2">
                                {this.props.title || 'Something went wrong'}
                            </h3>
                            <p className="text-red-600 mb-4">
                                {this.props.message || 'We encountered an unexpected error. Please try again.'}
                            </p>
                            <button
                                onClick={this.handleRetry}
                                className="inline-flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                </svg>
                                Try Again
                            </button>
                            {this.props.showDetails && this.state.error && (
                                <details className="mt-4 text-left">
                                    <summary className="cursor-pointer text-sm text-red-600 hover:text-red-800">
                                        Show error details
                                    </summary>
                                    <pre className="mt-2 p-3 bg-red-100 rounded text-xs overflow-x-auto text-red-800">
                                        {this.state.error.toString()}
                                        {this.state.errorInfo?.componentStack}
                                    </pre>
                                </details>
                            )}
                        </div>
                    );
                }

                return this.props.children;
            }
        }

        // Functional error display with retry (for async errors)
        function ErrorDisplay({ error, onRetry, title = 'Error', message }) {
            const displayMessage = message || error?.message || 'An unexpected error occurred. Please try again.';

            return (
                <div className="bg-red-50 border border-red-200 rounded-lg p-6">
                    <div className="flex items-start gap-4">
                        <div className="flex-shrink-0">
                            <div className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-red-100">
                                <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                </svg>
                            </div>
                        </div>
                        <div className="flex-grow">
                            <h3 className="text-lg font-semibold text-red-800 mb-1">{title}</h3>
                            <p className="text-red-600 mb-3">{displayMessage}</p>
                            {onRetry && (
                                <button
                                    onClick={onRetry}
                                    className="inline-flex items-center gap-2 px-4 py-2 bg-red-600 text-white text-sm rounded-md hover:bg-red-700 transition-colors"
                                >
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                    </svg>
                                    Try Again
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            );
        }

        // ==================== End Skeleton & Error Components ====================

        // Login form component
        function LoginForm({ onLogin }) {
            const [username, setUsername] = useState('');
            const [password, setPassword] = useState('');
            const [error, setError] = useState('');

            const handleSubmit = async (e) => {
                e.preventDefault();
                try {
                    const formData = new URLSearchParams();
                    formData.append('username', username);
                    formData.append('password', password);
                    const res = await fetch('/api/auth/token', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        body: formData
                    });
                    if (!res.ok) throw new Error('Invalid credentials');
                    const data = await res.json();
                    localStorage.setItem('token', data.access_token);
                    onLogin();
                } catch (err) {
                    setError(err.message);
                }
            };

            return (
                <div className="max-w-md mx-auto mt-20 p-6 bg-white rounded-lg shadow">
                    <h2 className="text-2xl font-bold mb-4">Login</h2>
                    {error && <p className="text-red-500 mb-4">{error}</p>}
                    <form onSubmit={handleSubmit}>
                        <input
                            type="text"
                            placeholder="Username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="w-full p-2 border rounded mb-4"
                        />
                        <input
                            type="password"
                            placeholder="Password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="w-full p-2 border rounded mb-4"
                        />
                        <button
                            type="submit"
                            className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
                        >
                            Login
                        </button>
                    </form>
                    <p className="mt-4 text-sm text-gray-600">
                        Note: Authentication is optional. You can browse the dashboard without logging in.
                    </p>
                </div>
            );
        }

        // Dashboard summary component
        function DashboardSummary({ summary }) {
            return (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                    <div className="bg-white p-4 rounded-lg shadow">
                        <h3 className="text-gray-500 text-sm">Total Agents</h3>
                        <p className="text-2xl font-bold">{summary.total_agents}</p>
                    </div>
                    <div className="bg-white p-4 rounded-lg shadow">
                        <h3 className="text-gray-500 text-sm">Test Suites</h3>
                        <p className="text-2xl font-bold">{summary.total_suites}</p>
                    </div>
                    <div className="bg-white p-4 rounded-lg shadow">
                        <h3 className="text-gray-500 text-sm">Total Executions</h3>
                        <p className="text-2xl font-bold">{summary.total_executions}</p>
                    </div>
                    <div className="bg-white p-4 rounded-lg shadow">
                        <h3 className="text-gray-500 text-sm">Recent Success Rate</h3>
                        <p className="text-2xl font-bold">
                            {(summary.recent_success_rate * 100).toFixed(1)}%
                        </p>
                    </div>
                </div>
            );
        }

        // Suite list component
        function SuiteList({ executions, onSelect }) {
            return (
                <div className="bg-white rounded-lg shadow overflow-hidden">
                    <table className="min-w-full">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Suite</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Agent</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Success Rate</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Started</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            {executions.map((exec) => (
                                <tr
                                    key={exec.id}
                                    className="hover:bg-gray-50 cursor-pointer"
                                    onClick={() => onSelect(exec.id)}
                                >
                                    <td className="px-4 py-3">{exec.suite_name}</td>
                                    <td className="px-4 py-3">{exec.agent_name || '-'}</td>
                                    <td className="px-4 py-3">
                                        <span className={`px-2 py-1 text-xs rounded ${
                                            exec.status === 'completed' ? 'bg-green-100 text-green-800' :
                                            exec.status === 'failed' ? 'bg-red-100 text-red-800' :
                                            'bg-yellow-100 text-yellow-800'
                                        }`}>
                                            {exec.status}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3">{(exec.success_rate * 100).toFixed(1)}%</td>
                                    <td className="px-4 py-3">
                                        {new Date(exec.started_at).toLocaleString()}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            );
        }

        // Suite detail component
        function SuiteDetail({ executionId, onBack }) {
            const [execution, setExecution] = useState(null);
            const [loading, setLoading] = useState(true);

            useEffect(() => {
                api.get(`/suites/${executionId}`)
                    .then(setExecution)
                    .finally(() => setLoading(false));
            }, [executionId]);

            if (loading) {
                return (
                    <div>
                        <SkeletonBox className="h-5 w-24 mb-4" />
                        <SkeletonBox className="h-7 w-48 mb-4" />
                        <div className="grid grid-cols-3 gap-4 mb-6">
                            {[1, 2, 3].map((i) => (
                                <div key={i} className="bg-white p-4 rounded shadow">
                                    <SkeletonBox className="h-4 w-16 mb-2" />
                                    <SkeletonBox className="h-6 w-24" />
                                </div>
                            ))}
                        </div>
                        <SkeletonBox className="h-6 w-16 mb-2" />
                        <SkeletonSuiteList rows={4} />
                    </div>
                );
            }
            if (!execution) {
                return (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
                        <p className="text-yellow-700">Execution not found. It may have been deleted.</p>
                        <button
                            onClick={onBack}
                            className="mt-4 text-blue-500 hover:text-blue-700"
                        >
                            &larr; Back to list
                        </button>
                    </div>
                );
            }

            return (
                <div>
                    <button
                        onClick={onBack}
                        className="mb-4 text-blue-500 hover:underline"
                    >
                        &larr; Back to list
                    </button>
                    <h2 className="text-xl font-bold mb-4">{execution.suite_name}</h2>
                    <div className="grid grid-cols-3 gap-4 mb-6">
                        <div className="bg-white p-4 rounded shadow">
                            <h3 className="text-gray-500 text-sm">Agent</h3>
                            <p className="font-bold">{execution.agent_name}</p>
                        </div>
                        <div className="bg-white p-4 rounded shadow">
                            <h3 className="text-gray-500 text-sm">Success Rate</h3>
                            <p className="font-bold">{(execution.success_rate * 100).toFixed(1)}%</p>
                        </div>
                        <div className="bg-white p-4 rounded shadow">
                            <h3 className="text-gray-500 text-sm">Duration</h3>
                            <p className="font-bold">
                                {execution.duration_seconds?.toFixed(2) || '-'}s
                            </p>
                        </div>
                    </div>
                    <h3 className="text-lg font-bold mb-2">Tests</h3>
                    <div className="bg-white rounded-lg shadow overflow-hidden">
                        <table className="min-w-full">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Test</th>
                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Score</th>
                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Runs</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-200">
                                {execution.tests.map((test) => (
                                    <tr key={test.id}>
                                        <td className="px-4 py-3">{test.test_name}</td>
                                        <td className="px-4 py-3">
                                            <span className={`px-2 py-1 text-xs rounded ${
                                                test.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                                            }`}>
                                                {test.success ? 'Passed' : 'Failed'}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            {test.score?.toFixed(1) || '-'}/100
                                        </td>
                                        <td className="px-4 py-3">
                                            {test.successful_runs}/{test.total_runs}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            );
        }

        // Event type colors for styling
        const EVENT_COLORS = {
            tool_call: { bg: 'bg-blue-50', border: 'border-blue-400', text: 'text-blue-700', icon: 'T' },
            llm_request: { bg: 'bg-green-50', border: 'border-green-400', text: 'text-green-700', icon: 'L' },
            reasoning: { bg: 'bg-amber-50', border: 'border-amber-400', text: 'text-amber-700', icon: 'R' },
            error: { bg: 'bg-red-50', border: 'border-red-400', text: 'text-red-700', icon: 'E' },
            progress: { bg: 'bg-purple-50', border: 'border-purple-400', text: 'text-purple-700', icon: 'P' },
        };

        // Agent Selector component - multi-select dropdown for selecting agents
        function AgentSelector({ agents, selectedAgents, onSelectionChange, maxAgents = 3 }) {
            const [isOpen, setIsOpen] = useState(false);

            const toggleAgent = (agentName) => {
                if (selectedAgents.includes(agentName)) {
                    onSelectionChange(selectedAgents.filter(a => a !== agentName));
                } else if (selectedAgents.length < maxAgents) {
                    onSelectionChange([...selectedAgents, agentName]);
                }
            };

            return (
                <div className="relative">
                    <button
                        type="button"
                        onClick={() => setIsOpen(!isOpen)}
                        className="w-full bg-white border border-gray-300 rounded-md px-4 py-2 text-left flex justify-between items-center hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                        <span className="truncate">
                            {selectedAgents.length === 0
                                ? 'Select agents to compare...'
                                : selectedAgents.join(', ')}
                        </span>
                        <svg className={`w-5 h-5 transition-transform ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                    </button>
                    {isOpen && (
                        <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
                            {agents.length === 0 ? (
                                <div className="px-4 py-2 text-gray-500">No agents available</div>
                            ) : (
                                agents.map((agent) => {
                                    const isSelected = selectedAgents.includes(agent.name);
                                    const isDisabled = !isSelected && selectedAgents.length >= maxAgents;
                                    return (
                                        <div
                                            key={agent.name}
                                            onClick={() => !isDisabled && toggleAgent(agent.name)}
                                            className={`px-4 py-2 cursor-pointer flex items-center ${
                                                isSelected ? 'bg-blue-50' : ''
                                            } ${isDisabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-100'}`}
                                        >
                                            <input
                                                type="checkbox"
                                                checked={isSelected}
                                                onChange={() => {}}
                                                disabled={isDisabled}
                                                className="mr-3"
                                            />
                                            <div>
                                                <div className="font-medium">{agent.name}</div>
                                                {agent.agent_type && (
                                                    <div className="text-sm text-gray-500">{agent.agent_type}</div>
                                                )}
                                            </div>
                                        </div>
                                    );
                                })
                            )}
                        </div>
                    )}
                    <div className="mt-1 text-xs text-gray-500">
                        {selectedAgents.length}/2-{maxAgents} agents selected
                    </div>
                </div>
            );
        }

        // Event Item component - displays a single event with type-specific styling
        function EventItem({ event, sequence }) {
            const [expanded, setExpanded] = useState(false);
            const colors = EVENT_COLORS[event.event_type] || {
                bg: 'bg-gray-50',
                border: 'border-gray-400',
                text: 'text-gray-700',
                icon: '?'
            };

            return (
                <div
                    className={`${colors.bg} border-l-4 ${colors.border} p-3 rounded-r mb-2 cursor-pointer hover:shadow-sm transition-shadow`}
                    onClick={() => setExpanded(!expanded)}
                >
                    <div className="flex items-start">
                        <div className={`w-6 h-6 rounded-full ${colors.border.replace('border', 'bg').replace('400', '200')} ${colors.text} flex items-center justify-center text-xs font-bold mr-3 flex-shrink-0`}>
                            {colors.icon}
                        </div>
                        <div className="flex-grow min-w-0">
                            <div className="flex items-center justify-between">
                                <span className={`text-xs font-semibold uppercase ${colors.text}`}>
                                    {event.event_type.replace('_', ' ')}
                                </span>
                                <span className="text-xs text-gray-400">#{sequence + 1}</span>
                            </div>
                            <p className="text-sm text-gray-700 mt-1 break-words">{event.summary}</p>
                            {expanded && event.data && (
                                <div className="mt-2 p-2 bg-white rounded text-xs font-mono overflow-x-auto">
                                    <pre className="whitespace-pre-wrap">{JSON.stringify(event.data, null, 2)}</pre>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            );
        }

        // ==================== Metrics Panel Component ====================

        // Helper function to find best values across agents
        function findBestMetrics(agents) {
            if (!agents || agents.length === 0) return {};

            const metrics = {
                score: { best: -Infinity, isBest: (a, b) => a > b },
                total_tokens: { best: Infinity, isBest: (a, b) => a < b },
                total_steps: { best: Infinity, isBest: (a, b) => a < b },
                duration_seconds: { best: Infinity, isBest: (a, b) => a < b },
                cost_usd: { best: Infinity, isBest: (a, b) => a < b },
            };

            agents.forEach(agent => {
                Object.keys(metrics).forEach(key => {
                    const value = agent[key];
                    if (value !== null && value !== undefined) {
                        if (metrics[key].isBest(value, metrics[key].best)) {
                            metrics[key].best = value;
                        }
                    }
                });
            });

            return metrics;
        }

        // Helper function to calculate percentage difference
        function calculatePercentDiff(value, baseline) {
            if (baseline === null || baseline === undefined || baseline === 0) return null;
            if (value === null || value === undefined) return null;
            return ((value - baseline) / Math.abs(baseline)) * 100;
        }

        // MetricValue component - displays a single metric with optional highlighting
        function MetricValue({ label, value, unit, isBest, isWorst, percentDiff, format }) {
            const formatValue = (val) => {
                if (val === null || val === undefined) return '-';
                if (format === 'number') return val.toLocaleString();
                if (format === 'decimal') return val.toFixed(2);
                if (format === 'score') return val.toFixed(1);
                if (format === 'cost') return val.toFixed(4);
                return val;
            };

            const getBgColor = () => {
                if (isBest) return 'bg-green-50';
                if (isWorst) return 'bg-red-50';
                return 'bg-white';
            };

            const getTextColor = () => {
                if (isBest) return 'text-green-700 font-semibold';
                if (isWorst) return 'text-red-700';
                return 'text-gray-700';
            };

            const getDiffColor = () => {
                if (percentDiff === null) return 'text-gray-400';
                if (percentDiff > 0) return label === 'Score' ? 'text-green-600' : 'text-red-600';
                if (percentDiff < 0) return label === 'Score' ? 'text-red-600' : 'text-green-600';
                return 'text-gray-500';
            };

            return (
                <div className={`p-2 rounded ${getBgColor()} border border-gray-100`}>
                    <div className="text-xs text-gray-500 mb-1">{label}</div>
                    <div className={`text-sm ${getTextColor()}`}>
                        {formatValue(value)}{unit && <span className="text-gray-400 ml-0.5">{unit}</span>}
                        {isBest && <span className="ml-1 text-green-600 text-xs">★</span>}
                    </div>
                    {percentDiff !== null && (
                        <div className={`text-xs ${getDiffColor()}`}>
                            {percentDiff > 0 ? '+' : ''}{percentDiff.toFixed(1)}%
                        </div>
                    )}
                </div>
            );
        }

        // MetricsPanel component - displays metrics comparison for multiple agents
        function MetricsPanel({ agents, showPercentDiff = true, baselineIndex = 0 }) {
            if (!agents || agents.length === 0) {
                return (
                    <div className="bg-gray-50 p-4 rounded text-center text-gray-500">
                        No agent data available for comparison
                    </div>
                );
            }

            const bestMetrics = findBestMetrics(agents);
            const baseline = agents[baselineIndex];

            // Determine worst values for highlighting
            const worstMetrics = {};
            if (agents.length > 1) {
                const metricKeys = ['score', 'total_tokens', 'total_steps', 'duration_seconds', 'cost_usd'];
                metricKeys.forEach(key => {
                    let worst = key === 'score' ? Infinity : -Infinity;
                    agents.forEach(agent => {
                        const value = agent[key];
                        if (value !== null && value !== undefined) {
                            if (key === 'score') {
                                if (value < worst) worst = value;
                            } else {
                                if (value > worst) worst = value;
                            }
                        }
                    });
                    worstMetrics[key] = worst;
                });
            }

            const isBest = (agent, key) => {
                const value = agent[key];
                if (value === null || value === undefined) return false;
                return value === bestMetrics[key]?.best;
            };

            const isWorst = (agent, key) => {
                if (agents.length <= 1) return false;
                const value = agent[key];
                if (value === null || value === undefined) return false;
                return value === worstMetrics[key] && !isBest(agent, key);
            };

            return (
                <div className="bg-white rounded-lg shadow">
                    <div className="p-4 border-b bg-gray-50 rounded-t-lg">
                        <h3 className="text-lg font-bold text-gray-800">Metrics Comparison</h3>
                        <p className="text-sm text-gray-500 mt-1">
                            {agents.length} agent{agents.length > 1 ? 's' : ''} compared
                            {showPercentDiff && agents.length > 1 && (
                                <span> • % diff vs {baseline.agent_name}</span>
                            )}
                        </p>
                    </div>

                    {/* Responsive grid layout */}
                    <div className={`grid gap-4 p-4 ${
                        agents.length === 1 ? 'grid-cols-1' :
                        agents.length === 2 ? 'grid-cols-1 md:grid-cols-2' :
                        'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'
                    }`}>
                        {agents.map((agent, idx) => (
                            <div key={agent.agent_name} className="border rounded-lg overflow-hidden">
                                {/* Agent header */}
                                <div className={`p-3 ${
                                    agent.success ? 'bg-green-50 border-b border-green-200' : 'bg-red-50 border-b border-red-200'
                                }`}>
                                    <div className="flex items-center justify-between">
                                        <h4 className="font-semibold text-gray-800">{agent.agent_name}</h4>
                                        <span className={`px-2 py-0.5 text-xs rounded-full ${
                                            agent.success
                                                ? 'bg-green-100 text-green-800'
                                                : 'bg-red-100 text-red-800'
                                        }`}>
                                            {agent.success ? 'Passed' : 'Failed'}
                                        </span>
                                    </div>
                                    {idx === baselineIndex && agents.length > 1 && (
                                        <span className="text-xs text-gray-500">Baseline</span>
                                    )}
                                </div>

                                {/* Metrics grid */}
                                <div className="grid grid-cols-2 gap-2 p-3">
                                    <MetricValue
                                        label="Score"
                                        value={agent.score}
                                        unit="/100"
                                        format="score"
                                        isBest={isBest(agent, 'score')}
                                        isWorst={isWorst(agent, 'score')}
                                        percentDiff={showPercentDiff && idx !== baselineIndex
                                            ? calculatePercentDiff(agent.score, baseline.score)
                                            : null}
                                    />
                                    <MetricValue
                                        label="Tokens"
                                        value={agent.total_tokens}
                                        format="number"
                                        isBest={isBest(agent, 'total_tokens')}
                                        isWorst={isWorst(agent, 'total_tokens')}
                                        percentDiff={showPercentDiff && idx !== baselineIndex
                                            ? calculatePercentDiff(agent.total_tokens, baseline.total_tokens)
                                            : null}
                                    />
                                    <MetricValue
                                        label="Steps"
                                        value={agent.total_steps}
                                        format="number"
                                        isBest={isBest(agent, 'total_steps')}
                                        isWorst={isWorst(agent, 'total_steps')}
                                        percentDiff={showPercentDiff && idx !== baselineIndex
                                            ? calculatePercentDiff(agent.total_steps, baseline.total_steps)
                                            : null}
                                    />
                                    <MetricValue
                                        label="Duration"
                                        value={agent.duration_seconds}
                                        unit="s"
                                        format="decimal"
                                        isBest={isBest(agent, 'duration_seconds')}
                                        isWorst={isWorst(agent, 'duration_seconds')}
                                        percentDiff={showPercentDiff && idx !== baselineIndex
                                            ? calculatePercentDiff(agent.duration_seconds, baseline.duration_seconds)
                                            : null}
                                    />
                                    <MetricValue
                                        label="Cost"
                                        value={agent.cost_usd}
                                        unit="$"
                                        format="cost"
                                        isBest={isBest(agent, 'cost_usd')}
                                        isWorst={isWorst(agent, 'cost_usd')}
                                        percentDiff={showPercentDiff && idx !== baselineIndex
                                            ? calculatePercentDiff(agent.cost_usd, baseline.cost_usd)
                                            : null}
                                    />
                                    <MetricValue
                                        label="Tool Calls"
                                        value={agent.tool_calls}
                                        format="number"
                                    />
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Legend */}
                    <div className="p-3 border-t bg-gray-50 rounded-b-lg flex flex-wrap gap-4 text-xs">
                        <div className="flex items-center gap-1">
                            <span className="w-3 h-3 rounded bg-green-50 border border-green-200"></span>
                            <span className="text-gray-600">Best value</span>
                            <span className="text-green-600 ml-0.5">★</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <span className="w-3 h-3 rounded bg-red-50 border border-red-200"></span>
                            <span className="text-gray-600">Worst value</span>
                        </div>
                        {showPercentDiff && agents.length > 1 && (
                            <div className="flex items-center gap-1">
                                <span className="text-gray-600">% shown relative to baseline</span>
                            </div>
                        )}
                    </div>
                </div>
            );
        }

        // ==================== End Metrics Panel Component ====================

        // Step Comparison component - displays event list for a single agent
        // StepComparison component - displays event list for a single agent with clickable events
        function StepComparison({ agentDetail }) {
            const [eventFilter, setEventFilter] = useState('all');
            const [selectedEvent, setSelectedEvent] = useState(null);

            if (!agentDetail) {
                return (
                    <div className="bg-gray-50 p-4 rounded text-center text-gray-500">
                        No execution data available
                    </div>
                );
            }

            // Calculate event counts for filter buttons
            const eventCounts = (agentDetail.events || []).reduce((acc, e) => {
                acc[e.event_type] = (acc[e.event_type] || 0) + 1;
                acc.all = (acc.all || 0) + 1;
                return acc;
            }, { all: 0 });

            const filteredEvents = eventFilter === 'all'
                ? agentDetail.events
                : agentDetail.events.filter(e => e.event_type === eventFilter);

            return (
                <div className="bg-white rounded-lg shadow">
                    {/* Agent header with metrics */}
                    <div className="p-4 border-b bg-gray-50 rounded-t-lg">
                        <h4 className="font-bold text-lg mb-2">{agentDetail.agent_name}</h4>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                            <div>
                                <span className="text-gray-500">Score:</span>
                                <span className={`ml-2 font-semibold ${agentDetail.success ? 'text-green-600' : 'text-red-600'}`}>
                                    {agentDetail.score?.toFixed(1) || '-'}/100
                                </span>
                            </div>
                            <div>
                                <span className="text-gray-500">Status:</span>
                                <span className={`ml-2 px-2 py-0.5 text-xs rounded ${
                                    agentDetail.success
                                        ? 'bg-green-100 text-green-800'
                                        : 'bg-red-100 text-red-800'
                                }`}>
                                    {agentDetail.success ? 'Passed' : 'Failed'}
                                </span>
                            </div>
                            <div>
                                <span className="text-gray-500">Duration:</span>
                                <span className="ml-2">{agentDetail.duration_seconds?.toFixed(2) || '-'}s</span>
                            </div>
                            <div>
                                <span className="text-gray-500">Tokens:</span>
                                <span className="ml-2">{agentDetail.total_tokens?.toLocaleString() || '-'}</span>
                            </div>
                            <div>
                                <span className="text-gray-500">Steps:</span>
                                <span className="ml-2">{agentDetail.total_steps || '-'}</span>
                            </div>
                            <div>
                                <span className="text-gray-500">Cost:</span>
                                <span className="ml-2">${agentDetail.cost_usd?.toFixed(4) || '-'}</span>
                            </div>
                        </div>
                    </div>

                    {/* Event filters using EventFilters component */}
                    <div className="p-2 border-b">
                        <EventFilters
                            eventFilter={eventFilter}
                            onFilterChange={setEventFilter}
                            eventCounts={eventCounts}
                            showCounts={true}
                        />
                    </div>

                    {/* Event list with click to open detail panel */}
                    <div className="p-4 max-h-96 overflow-y-auto">
                        {filteredEvents && filteredEvents.length > 0 ? (
                            filteredEvents.map((event, idx) => (
                                <div
                                    key={idx}
                                    onClick={() => setSelectedEvent({
                                        ...event,
                                        relative_time_ms: 0,
                                        duration_ms: null
                                    })}
                                    className="cursor-pointer"
                                >
                                    <EventItem event={event} sequence={event.sequence} />
                                </div>
                            ))
                        ) : (
                            <p className="text-gray-500 text-center py-4">No events to display</p>
                        )}
                    </div>

                    {/* Event detail panel */}
                    {selectedEvent && (
                        <EventDetailPanel
                            event={selectedEvent}
                            onClose={() => setSelectedEvent(null)}
                        />
                    )}
                </div>
            );
        }

        // Comparison Container component - main layout with columns for side-by-side comparison
        function ComparisonContainer({ suiteName, testId, agents, availableAgents, onTestSelect, tests }) {
            const [comparison, setComparison] = useState(null);
            const [loading, setLoading] = useState(false);
            const [error, setError] = useState(null);
            const [selectedAgents, setSelectedAgents] = useState([]);

            const loadComparison = useCallback(async () => {
                if (selectedAgents.length < 2 || !testId) {
                    setError('Please select a test and at least 2 agents');
                    return;
                }
                setLoading(true);
                setError(null);
                try {
                    const agentParams = selectedAgents.map(a => `agents=${encodeURIComponent(a)}`).join('&');
                    const data = await api.get(
                        `/compare/side-by-side?suite_name=${encodeURIComponent(suiteName)}&test_id=${encodeURIComponent(testId)}&${agentParams}`
                    );
                    setComparison(data);
                } catch (err) {
                    setError(err.message || 'Failed to load comparison data');
                    setComparison(null);
                } finally {
                    setLoading(false);
                }
            }, [suiteName, testId, selectedAgents]);

            // Loading state
            if (loading) {
                return (
                    <div>
                        <SkeletonMetricsPanel agents={selectedAgents.length || 2} />
                        <div className="mt-4">
                            <SkeletonBox className="h-5 w-40 mb-2" />
                            <div className={`grid gap-4 ${
                                selectedAgents.length === 2 ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1 lg:grid-cols-3'
                            }`}>
                                {Array.from({ length: selectedAgents.length || 2 }).map((_, i) => (
                                    <div key={i} className="bg-white rounded-lg shadow p-4">
                                        <SkeletonBox className="h-6 w-32 mb-4" />
                                        {[1, 2, 3, 4].map((j) => (
                                            <div key={j} className="mb-2">
                                                <SkeletonBox className="h-16 w-full rounded" />
                                            </div>
                                        ))}
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                );
            }

            // Selection UI
            if (!comparison) {
                return (
                    <div className="bg-white p-6 rounded-lg shadow">
                        <h3 className="text-lg font-bold mb-4">Step-by-Step Comparison</h3>

                        {error && (
                            <ErrorDisplay
                                error={{ message: error }}
                                onRetry={loadComparison}
                                title="Failed to load comparison"
                            />
                        )}

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Select Test:
                                </label>
                                <select
                                    value={testId || ''}
                                    onChange={(e) => onTestSelect(e.target.value)}
                                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                >
                                    <option value="">Choose a test...</option>
                                    {tests && tests.map((test) => (
                                        <option key={test.test_id} value={test.test_id}>
                                            {test.test_name || test.test_id}
                                        </option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Select Agents (2-3):
                                </label>
                                <AgentSelector
                                    agents={availableAgents}
                                    selectedAgents={selectedAgents}
                                    onSelectionChange={setSelectedAgents}
                                    maxAgents={3}
                                />
                            </div>
                        </div>

                        <button
                            onClick={loadComparison}
                            disabled={selectedAgents.length < 2 || !testId}
                            className="w-full md:w-auto bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                        >
                            Compare Agents
                        </button>

                        <p className="mt-4 text-sm text-gray-500">
                            Select a test and 2-3 agents to view their step-by-step execution comparison.
                        </p>
                    </div>
                );
            }

            // Comparison results view
            return (
                <div>
                    {/* Header with test info and reset button */}
                    <div className="bg-white p-4 rounded-lg shadow mb-4 flex flex-wrap items-center justify-between gap-4">
                        <div>
                            <h3 className="text-lg font-bold">{comparison.test_name}</h3>
                            <p className="text-sm text-gray-500">
                                Suite: {comparison.suite_name} | Test ID: {comparison.test_id}
                            </p>
                        </div>
                        <button
                            onClick={() => setComparison(null)}
                            className="text-blue-500 hover:text-blue-700 flex items-center gap-1"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                            </svg>
                            Back to selection
                        </button>
                    </div>

                    {/* Metrics Comparison Panel */}
                    {comparison.agents.length > 0 && (
                        <div className="mb-4">
                            <MetricsPanel agents={comparison.agents} showPercentDiff={true} baselineIndex={0} />
                        </div>
                    )}

                    {/* Side-by-side step comparison columns */}
                    <h4 className="text-md font-semibold text-gray-700 mb-2">Step-by-Step Events</h4>
                    <div className={`grid gap-4 ${
                        comparison.agents.length === 2
                            ? 'grid-cols-1 lg:grid-cols-2'
                            : 'grid-cols-1 lg:grid-cols-3'
                    }`}>
                        {comparison.agents.map((agent) => (
                            <StepComparison key={agent.agent_name} agentDetail={agent} />
                        ))}
                    </div>

                    {comparison.agents.length === 0 && (
                        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-yellow-700">
                            No execution data found for the selected agents on this test.
                        </div>
                    )}
                </div>
            );
        }

        // Agent comparison component (original - for high-level metrics)
        function AgentComparison({ suiteName, agents }) {
            const [comparison, setComparison] = useState(null);
            const [selectedAgents, setSelectedAgents] = useState([]);
            const [loading, setLoading] = useState(false);
            const [tests, setTests] = useState([]);
            const [selectedTestId, setSelectedTestId] = useState('');
            const [viewMode, setViewMode] = useState('metrics'); // 'metrics' or 'steps'

            // Load tests for the suite
            useEffect(() => {
                if (suiteName) {
                    api.get(`/suites?suite_name=${encodeURIComponent(suiteName)}&limit=1`)
                        .then((data) => {
                            if (data.items && data.items.length > 0) {
                                const executionId = data.items[0].id;
                                return api.get(`/suites/${executionId}`);
                            }
                            return null;
                        })
                        .then((detail) => {
                            if (detail && detail.tests) {
                                setTests(detail.tests);
                            }
                        })
                        .catch(console.error);
                }
            }, [suiteName]);

            const loadComparison = useCallback(async () => {
                if (selectedAgents.length < 2) return;
                setLoading(true);
                try {
                    const agentParams = selectedAgents.map(a => `agents=${encodeURIComponent(a)}`).join('&');
                    const data = await api.get(`/compare/agents?suite_name=${encodeURIComponent(suiteName)}&${agentParams}`);
                    setComparison(data);
                } finally {
                    setLoading(false);
                }
            }, [suiteName, selectedAgents]);

            // Step comparison view
            if (viewMode === 'steps') {
                return (
                    <div>
                        <div className="mb-4 flex items-center gap-4">
                            <button
                                onClick={() => setViewMode('metrics')}
                                className="text-blue-500 hover:text-blue-700 flex items-center gap-1"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                                </svg>
                                Back to metrics
                            </button>
                        </div>
                        <ComparisonContainer
                            suiteName={suiteName}
                            testId={selectedTestId}
                            agents={selectedAgents}
                            availableAgents={agents}
                            onTestSelect={setSelectedTestId}
                            tests={tests}
                        />
                    </div>
                );
            }

            // Metrics comparison view (original)
            if (!comparison) {
                return (
                    <div className="bg-white p-4 rounded shadow">
                        <h3 className="font-bold mb-4">Agent Comparison</h3>
                        <p className="text-gray-600 mb-4">
                            Select agents to compare (requires at least 2 agents and test data)
                        </p>
                        <div className="flex flex-wrap gap-2 mb-4">
                            {agents.map((agent) => (
                                <label key={agent.name} className="flex items-center">
                                    <input
                                        type="checkbox"
                                        checked={selectedAgents.includes(agent.name)}
                                        onChange={(e) => {
                                            if (e.target.checked) {
                                                setSelectedAgents([...selectedAgents, agent.name]);
                                            } else {
                                                setSelectedAgents(selectedAgents.filter(a => a !== agent.name));
                                            }
                                        }}
                                        className="mr-2"
                                    />
                                    {agent.name}
                                </label>
                            ))}
                        </div>
                        <button
                            onClick={loadComparison}
                            disabled={selectedAgents.length < 2 || loading}
                            className="bg-blue-500 text-white px-4 py-2 rounded disabled:bg-gray-300"
                        >
                            {loading ? 'Loading...' : 'Compare'}
                        </button>
                    </div>
                );
            }

            return (
                <div className="bg-white p-4 rounded shadow">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="font-bold">Agent Comparison: {comparison.suite_name}</h3>
                        <button
                            onClick={() => setViewMode('steps')}
                            className="bg-green-500 text-white px-4 py-2 rounded text-sm hover:bg-green-600"
                        >
                            View Step-by-Step
                        </button>
                    </div>
                    <table className="min-w-full mb-4">
                        <thead>
                            <tr>
                                <th className="px-4 py-2 text-left">Agent</th>
                                <th className="px-4 py-2 text-left">Executions</th>
                                <th className="px-4 py-2 text-left">Avg Success</th>
                                <th className="px-4 py-2 text-left">Avg Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {comparison.agents.map((agent) => (
                                <tr key={agent.agent_name}>
                                    <td className="px-4 py-2">{agent.agent_name}</td>
                                    <td className="px-4 py-2">{agent.total_executions}</td>
                                    <td className="px-4 py-2">{(agent.avg_success_rate * 100).toFixed(1)}%</td>
                                    <td className="px-4 py-2">{agent.avg_score?.toFixed(1) || '-'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <button
                        onClick={() => setComparison(null)}
                        className="text-blue-500 hover:underline"
                    >
                        Reset
                    </button>
                </div>
            );
        }

        // ==================== Timeline Components ====================

        // TimeScale component - renders time axis with labels
        function TimeScale({ totalDurationMs, zoomLevel, width }) {
            // Calculate tick intervals based on duration and zoom
            const effectiveDuration = totalDurationMs / zoomLevel;
            const tickCount = Math.min(10, Math.max(4, Math.floor(width / 100)));
            const tickInterval = effectiveDuration / tickCount;

            const ticks = [];
            for (let i = 0; i <= tickCount; i++) {
                const timeMs = i * tickInterval;
                const position = (i / tickCount) * 100;
                ticks.push({ timeMs, position });
            }

            const formatTime = (ms) => {
                if (ms >= 60000) {
                    const mins = Math.floor(ms / 60000);
                    const secs = Math.floor((ms % 60000) / 1000);
                    return `${mins}m ${secs}s`;
                } else if (ms >= 1000) {
                    return `${(ms / 1000).toFixed(1)}s`;
                }
                return `${Math.round(ms)}ms`;
            };

            return (
                <div className="relative h-8 border-b border-gray-300 bg-gray-50">
                    {ticks.map((tick, idx) => (
                        <div
                            key={idx}
                            className="absolute top-0 h-full flex flex-col items-center"
                            style={{ left: `${tick.position}%`, transform: 'translateX(-50%)' }}
                        >
                            <div className="h-2 w-px bg-gray-400"></div>
                            <span className="text-xs text-gray-500 mt-1 whitespace-nowrap">
                                {formatTime(tick.timeMs)}
                            </span>
                        </div>
                    ))}
                </div>
            );
        }

        // EventMarker component - individual event on timeline with type-based colors
        function EventMarker({ event, totalDurationMs, zoomLevel, onHover, onLeave, onClick }) {
            const colors = EVENT_COLORS[event.event_type] || {
                bg: 'bg-gray-400',
                border: 'border-gray-500',
                text: 'text-gray-700',
                icon: '?'
            };

            // Calculate position based on relative time
            const position = (event.relative_time_ms / totalDurationMs) * 100 * zoomLevel;

            // Calculate width based on duration (if available)
            const width = event.duration_ms
                ? Math.max(8, (event.duration_ms / totalDurationMs) * 100 * zoomLevel)
                : 8;

            // Clamp position to visible area
            if (position > 100) return null;

            return (
                <div
                    className={`absolute top-1/2 -translate-y-1/2 h-6 rounded cursor-pointer transition-all hover:ring-2 hover:ring-offset-1 ${colors.border.replace('border', 'bg').replace('400', '500')} hover:${colors.border.replace('border', 'bg').replace('400', '600')}`}
                    style={{
                        left: `${Math.min(position, 100 - (width / 10))}%`,
                        width: `${Math.max(width, 0.5)}%`,
                        minWidth: '8px',
                    }}
                    onMouseEnter={() => onHover(event)}
                    onMouseLeave={onLeave}
                    onClick={() => onClick(event)}
                    title={event.summary}
                >
                    <div className="flex items-center justify-center h-full text-white text-xs font-bold">
                        {colors.icon}
                    </div>
                </div>
            );
        }

        // EventTooltip component - hover tooltip with event summary
        function EventTooltip({ event, position }) {
            if (!event) return null;

            const colors = EVENT_COLORS[event.event_type] || {
                bg: 'bg-gray-50',
                border: 'border-gray-400',
                text: 'text-gray-700',
                icon: '?'
            };

            const formatTime = (ms) => {
                if (ms >= 60000) {
                    const mins = Math.floor(ms / 60000);
                    const secs = Math.floor((ms % 60000) / 1000);
                    return `${mins}m ${secs}s`;
                } else if (ms >= 1000) {
                    return `${(ms / 1000).toFixed(1)}s`;
                }
                return `${Math.round(ms)}ms`;
            };

            return (
                <div
                    className="absolute z-50 bg-white border border-gray-300 rounded-lg shadow-lg p-3 max-w-xs pointer-events-none"
                    style={{
                        left: `${Math.min(position.x, 70)}%`,
                        top: '100%',
                        marginTop: '8px',
                    }}
                >
                    <div className={`flex items-center gap-2 mb-2 ${colors.text}`}>
                        <span className={`w-5 h-5 rounded-full ${colors.border.replace('border', 'bg').replace('400', '200')} flex items-center justify-center text-xs font-bold`}>
                            {colors.icon}
                        </span>
                        <span className="font-semibold uppercase text-sm">
                            {event.event_type.replace('_', ' ')}
                        </span>
                        <span className="text-gray-400 text-xs">#{event.sequence + 1}</span>
                    </div>
                    <p className="text-sm text-gray-700 mb-2">{event.summary}</p>
                    <div className="flex gap-4 text-xs text-gray-500">
                        <span>Start: {formatTime(event.relative_time_ms)}</span>
                        {event.duration_ms && <span>Duration: {formatTime(event.duration_ms)}</span>}
                    </div>
                </div>
            );
        }

        // TimelineRow component - single agent timeline visualization with keyboard navigation
        function TimelineRow({ timeline, totalDurationMs, zoomLevel, eventFilter, onEventClick, rowIndex = 0 }) {
            const [hoveredEvent, setHoveredEvent] = useState(null);
            const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
            const [focusedIndex, setFocusedIndex] = useState(-1);
            const rowRef = React.useRef(null);
            const eventRefs = React.useRef([]);

            const filteredEvents = eventFilter === 'all'
                ? timeline.events
                : timeline.events.filter(e => e.event_type === eventFilter);

            const handleHover = (event) => {
                if (rowRef.current) {
                    const rect = rowRef.current.getBoundingClientRect();
                    const position = (event.relative_time_ms / totalDurationMs) * 100 * zoomLevel;
                    setHoverPosition({ x: position, y: 0 });
                }
                setHoveredEvent(event);
            };

            // Keyboard navigation handler
            const handleKeyDown = useCallback((e, idx) => {
                switch (e.key) {
                    case 'ArrowLeft':
                        e.preventDefault();
                        if (idx > 0) {
                            setFocusedIndex(idx - 1);
                            eventRefs.current[idx - 1]?.focus();
                        }
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        if (idx < filteredEvents.length - 1) {
                            setFocusedIndex(idx + 1);
                            eventRefs.current[idx + 1]?.focus();
                        }
                        break;
                    case 'Enter':
                    case ' ':
                        e.preventDefault();
                        onEventClick(filteredEvents[idx]);
                        break;
                    case 'Home':
                        e.preventDefault();
                        setFocusedIndex(0);
                        eventRefs.current[0]?.focus();
                        break;
                    case 'End':
                        e.preventDefault();
                        const lastIdx = filteredEvents.length - 1;
                        setFocusedIndex(lastIdx);
                        eventRefs.current[lastIdx]?.focus();
                        break;
                }
            }, [filteredEvents, onEventClick]);

            return (
                <div className="mb-4 last:mb-0" role="region" aria-label={`Timeline for ${timeline.agent_name}`}>
                    {/* Agent header */}
                    <div className="flex items-center justify-between mb-2 px-2">
                        <div className="flex items-center gap-2">
                            <span className="font-semibold text-gray-800">{timeline.agent_name}</span>
                            <span className="text-xs text-gray-500">
                                ({timeline.events.length} events)
                            </span>
                        </div>
                        <span className="text-xs text-gray-500">
                            Duration: {(timeline.total_duration_ms / 1000).toFixed(2)}s
                        </span>
                    </div>

                    {/* Timeline track with keyboard nav hint */}
                    <div className="relative">
                        <div
                            ref={rowRef}
                            className="relative h-10 bg-gray-100 rounded border border-gray-200 overflow-hidden"
                            role="listbox"
                            aria-label={`Events for ${timeline.agent_name}`}
                        >
                            {filteredEvents.map((event, idx) => {
                                const colors = EVENT_COLORS[event.event_type] || {
                                    bg: 'bg-gray-400',
                                    border: 'border-gray-500',
                                    text: 'text-gray-700',
                                    icon: '?'
                                };
                                const position = (event.relative_time_ms / totalDurationMs) * 100 * zoomLevel;
                                const width = event.duration_ms
                                    ? Math.max(8, (event.duration_ms / totalDurationMs) * 100 * zoomLevel)
                                    : 8;
                                if (position > 100) return null;

                                return (
                                    <button
                                        key={`${event.sequence}-${idx}`}
                                        ref={el => eventRefs.current[idx] = el}
                                        type="button"
                                        role="option"
                                        aria-selected={focusedIndex === idx}
                                        aria-label={`${event.event_type.replace('_', ' ')} event ${idx + 1}: ${event.summary}`}
                                        tabIndex={idx === 0 ? 0 : -1}
                                        className={`timeline-event-focusable absolute top-1/2 -translate-y-1/2 h-6 rounded cursor-pointer transition-all hover:ring-2 hover:ring-offset-1 ${colors.border.replace('border', 'bg').replace('400', '500')} hover:${colors.border.replace('border', 'bg').replace('400', '600')}`}
                                        style={{
                                            left: `${Math.min(position, 100 - (width / 10))}%`,
                                            width: `${Math.max(width, 0.5)}%`,
                                            minWidth: '8px',
                                        }}
                                        onMouseEnter={() => handleHover(event)}
                                        onMouseLeave={() => setHoveredEvent(null)}
                                        onFocus={() => {
                                            setFocusedIndex(idx);
                                            handleHover(event);
                                        }}
                                        onBlur={() => setHoveredEvent(null)}
                                        onClick={() => onEventClick(event)}
                                        onKeyDown={(e) => handleKeyDown(e, idx)}
                                        title={event.summary}
                                    >
                                        <div className="flex items-center justify-center h-full text-white text-xs font-bold">
                                            {colors.icon}
                                        </div>
                                    </button>
                                );
                            })}
                            {hoveredEvent && (
                                <EventTooltip event={hoveredEvent} position={hoverPosition} />
                            )}
                        </div>
                        {/* Keyboard navigation hint - only show on first row */}
                        {rowIndex === 0 && filteredEvents.length > 0 && (
                            <div className="absolute -bottom-5 left-0 text-xs text-gray-400">
                                Use arrow keys to navigate events, Enter to view details
                            </div>
                        )}
                    </div>
                </div>
            );
        }

        // ==================== EventFilters Component ====================
        // Toggle buttons for filtering events by type
        function EventFilters({ eventFilter, onFilterChange, eventCounts, showCounts = true }) {
            return (
                <div className="flex flex-wrap gap-1 items-center">
                    <span className="text-sm text-gray-600 mr-2">Filter:</span>
                    <button
                        onClick={() => onFilterChange('all')}
                        className={`px-3 py-1.5 text-xs rounded-full transition-all ${
                            eventFilter === 'all'
                                ? 'bg-gray-800 text-white shadow-sm'
                                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                        }`}
                    >
                        All {showCounts && eventCounts.all > 0 && `(${eventCounts.all})`}
                    </button>
                    {Object.entries(EVENT_COLORS).map(([type, colors]) => {
                        const count = eventCounts[type] || 0;
                        if (count === 0 && showCounts) return null;
                        return (
                            <button
                                key={type}
                                onClick={() => onFilterChange(type)}
                                className={`px-3 py-1.5 text-xs rounded-full transition-all flex items-center gap-1 ${
                                    eventFilter === type
                                        ? `${colors.bg} ${colors.text} ring-2 ring-offset-1 shadow-sm`
                                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                }`}
                            >
                                <span className={`w-4 h-4 rounded-full ${colors.border.replace('border', 'bg').replace('400', '400')} flex items-center justify-center text-white text-xs font-bold`}>
                                    {colors.icon}
                                </span>
                                <span>{type.replace('_', ' ')}</span>
                                {showCounts && count > 0 && <span>({count})</span>}
                            </button>
                        );
                    })}
                </div>
            );
        }

        // ==================== EventDetailPanel Component ====================
        // Full event details panel with type-specific displays

        // Helper component for tool_call details
        function ToolCallDetails({ data }) {
            const toolName = data.tool || data.tool_name || data.name || 'Unknown Tool';
            const args = data.arguments || data.args || data.input || data.parameters || {};
            const result = data.result || data.output || data.response || null;
            const status = data.status || (data.error ? 'error' : 'success');

            return (
                <div className="space-y-4">
                    {/* Tool Name */}
                    <div>
                        <h4 className="text-sm font-semibold text-gray-500 mb-2">Tool Name</h4>
                        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                            <code className="text-blue-800 font-mono text-sm font-semibold">{toolName}</code>
                            <span className={`ml-3 px-2 py-0.5 text-xs rounded-full ${
                                status === 'success' ? 'bg-green-100 text-green-700' :
                                status === 'error' ? 'bg-red-100 text-red-700' :
                                'bg-gray-100 text-gray-700'
                            }`}>
                                {status}
                            </span>
                        </div>
                    </div>

                    {/* Arguments */}
                    <div>
                        <h4 className="text-sm font-semibold text-gray-500 mb-2">Arguments</h4>
                        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 overflow-x-auto">
                            <pre className="text-xs font-mono whitespace-pre-wrap text-gray-700">
                                {typeof args === 'object' ? JSON.stringify(args, null, 2) : String(args)}
                            </pre>
                        </div>
                    </div>

                    {/* Result */}
                    {result !== null && (
                        <div>
                            <h4 className="text-sm font-semibold text-gray-500 mb-2">Result</h4>
                            <div className={`border rounded-lg p-3 overflow-x-auto ${
                                status === 'error' ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'
                            }`}>
                                <pre className={`text-xs font-mono whitespace-pre-wrap ${
                                    status === 'error' ? 'text-red-700' : 'text-green-700'
                                }`}>
                                    {typeof result === 'object' ? JSON.stringify(result, null, 2) : String(result)}
                                </pre>
                            </div>
                        </div>
                    )}
                </div>
            );
        }

        // Helper component for llm_request details
        function LLMRequestDetails({ data }) {
            const model = data.model || data.model_name || 'Unknown Model';
            const prompt = data.prompt || data.input || data.messages || data.request || '';
            const response = data.response || data.output || data.completion || data.content || '';
            const inputTokens = data.input_tokens || data.prompt_tokens || data.tokens?.input || null;
            const outputTokens = data.output_tokens || data.completion_tokens || data.tokens?.output || null;
            const totalTokens = data.total_tokens || data.tokens?.total || (inputTokens && outputTokens ? inputTokens + outputTokens : null);
            const cost = data.cost || data.cost_usd || null;

            // Truncate long text with show more/less
            const [showFullPrompt, setShowFullPrompt] = useState(false);
            const [showFullResponse, setShowFullResponse] = useState(false);

            const truncateText = (text, maxLength = 500) => {
                const textStr = typeof text === 'object' ? JSON.stringify(text, null, 2) : String(text);
                if (textStr.length <= maxLength) return { text: textStr, truncated: false };
                return { text: textStr.slice(0, maxLength) + '...', truncated: true, full: textStr };
            };

            const promptData = truncateText(prompt);
            const responseData = truncateText(response);

            return (
                <div className="space-y-4">
                    {/* Model and Token Info */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                            <div className="text-xs text-green-600 font-semibold mb-1">Model</div>
                            <div className="text-sm font-mono text-green-800">{model}</div>
                        </div>
                        {inputTokens !== null && (
                            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                                <div className="text-xs text-blue-600 font-semibold mb-1">Input Tokens</div>
                                <div className="text-sm font-mono text-blue-800">{inputTokens.toLocaleString()}</div>
                            </div>
                        )}
                        {outputTokens !== null && (
                            <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
                                <div className="text-xs text-purple-600 font-semibold mb-1">Output Tokens</div>
                                <div className="text-sm font-mono text-purple-800">{outputTokens.toLocaleString()}</div>
                            </div>
                        )}
                        {totalTokens !== null && (
                            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                                <div className="text-xs text-amber-600 font-semibold mb-1">Total Tokens</div>
                                <div className="text-sm font-mono text-amber-800">{totalTokens.toLocaleString()}</div>
                            </div>
                        )}
                        {cost !== null && (
                            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                                <div className="text-xs text-gray-600 font-semibold mb-1">Cost</div>
                                <div className="text-sm font-mono text-gray-800">${cost.toFixed(4)}</div>
                            </div>
                        )}
                    </div>

                    {/* Prompt */}
                    {prompt && (
                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <h4 className="text-sm font-semibold text-gray-500">Prompt</h4>
                                {promptData.truncated && (
                                    <button
                                        onClick={() => setShowFullPrompt(!showFullPrompt)}
                                        className="text-xs text-blue-600 hover:text-blue-800"
                                    >
                                        {showFullPrompt ? 'Show less' : 'Show full'}
                                    </button>
                                )}
                            </div>
                            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 overflow-x-auto max-h-60 overflow-y-auto">
                                <pre className="text-xs font-mono whitespace-pre-wrap text-blue-800">
                                    {showFullPrompt ? promptData.full || promptData.text : promptData.text}
                                </pre>
                            </div>
                        </div>
                    )}

                    {/* Response */}
                    {response && (
                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <h4 className="text-sm font-semibold text-gray-500">Response</h4>
                                {responseData.truncated && (
                                    <button
                                        onClick={() => setShowFullResponse(!showFullResponse)}
                                        className="text-xs text-green-600 hover:text-green-800"
                                    >
                                        {showFullResponse ? 'Show less' : 'Show full'}
                                    </button>
                                )}
                            </div>
                            <div className="bg-green-50 border border-green-200 rounded-lg p-3 overflow-x-auto max-h-60 overflow-y-auto">
                                <pre className="text-xs font-mono whitespace-pre-wrap text-green-800">
                                    {showFullResponse ? responseData.full || responseData.text : responseData.text}
                                </pre>
                            </div>
                        </div>
                    )}
                </div>
            );
        }

        // Helper component for error details
        function ErrorDetails({ data }) {
            const errorType = data.error_type || data.type || data.name || 'Error';
            const message = data.message || data.error || data.error_message || data.description || 'Unknown error';
            const stackTrace = data.stack_trace || data.stacktrace || data.stack || data.traceback || null;
            const code = data.code || data.error_code || null;

            return (
                <div className="space-y-4">
                    {/* Error Type and Code */}
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                        <div className="flex items-center gap-3 mb-2">
                            <span className="w-8 h-8 bg-red-200 rounded-full flex items-center justify-center">
                                <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                </svg>
                            </span>
                            <div>
                                <span className="font-semibold text-red-800">{errorType}</span>
                                {code && <span className="ml-2 text-xs bg-red-200 text-red-800 px-2 py-0.5 rounded">Code: {code}</span>}
                            </div>
                        </div>
                    </div>

                    {/* Error Message */}
                    <div>
                        <h4 className="text-sm font-semibold text-gray-500 mb-2">Error Message</h4>
                        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                            <p className="text-sm text-red-700 font-mono">{message}</p>
                        </div>
                    </div>

                    {/* Stack Trace */}
                    {stackTrace && (
                        <div>
                            <h4 className="text-sm font-semibold text-gray-500 mb-2">Stack Trace</h4>
                            <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 overflow-x-auto max-h-80 overflow-y-auto">
                                <pre className="text-xs font-mono whitespace-pre-wrap text-gray-300">
                                    {typeof stackTrace === 'object' ? JSON.stringify(stackTrace, null, 2) : String(stackTrace)}
                                </pre>
                            </div>
                        </div>
                    )}
                </div>
            );
        }

        // Helper component for reasoning details
        function ReasoningDetails({ data }) {
            const thought = data.thought || data.reasoning || data.thinking || data.content || '';
            const step = data.step || data.step_number || null;

            return (
                <div className="space-y-4">
                    {step !== null && (
                        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                            <span className="text-xs text-amber-600 font-semibold">Step {step}</span>
                        </div>
                    )}
                    <div>
                        <h4 className="text-sm font-semibold text-gray-500 mb-2">Reasoning</h4>
                        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 overflow-x-auto">
                            <pre className="text-xs font-mono whitespace-pre-wrap text-amber-800">
                                {typeof thought === 'object' ? JSON.stringify(thought, null, 2) : String(thought)}
                            </pre>
                        </div>
                    </div>
                </div>
            );
        }

        // Helper component for progress details
        function ProgressDetails({ data }) {
            const percentage = data.percentage || data.progress || data.percent || 0;
            const message = data.message || data.status || data.description || '';
            const current = data.current || null;
            const total = data.total || null;

            return (
                <div className="space-y-4">
                    {/* Progress Bar */}
                    <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-semibold text-purple-700">Progress</span>
                            <span className="text-sm font-mono text-purple-800">{percentage}%</span>
                        </div>
                        <div className="w-full bg-purple-200 rounded-full h-3">
                            <div
                                className="bg-purple-600 h-3 rounded-full transition-all"
                                style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
                            ></div>
                        </div>
                        {current !== null && total !== null && (
                            <div className="text-xs text-purple-600 mt-1">{current} / {total}</div>
                        )}
                    </div>

                    {/* Message */}
                    {message && (
                        <div>
                            <h4 className="text-sm font-semibold text-gray-500 mb-2">Status Message</h4>
                            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                                <p className="text-sm text-gray-700">{message}</p>
                            </div>
                        </div>
                    )}
                </div>
            );
        }

        // EventDetailPanel component - full event details panel with type-specific displays
        function EventDetailPanel({ event, onClose }) {
            if (!event) return null;

            const [copySuccess, setCopySuccess] = useState(false);

            const colors = EVENT_COLORS[event.event_type] || {
                bg: 'bg-gray-50',
                border: 'border-gray-400',
                text: 'text-gray-700',
                icon: '?'
            };

            const formatTime = (ms) => {
                if (ms >= 60000) {
                    const mins = Math.floor(ms / 60000);
                    const secs = Math.floor((ms % 60000) / 1000);
                    return `${mins}m ${secs}s`;
                } else if (ms >= 1000) {
                    return `${(ms / 1000).toFixed(1)}s`;
                }
                return `${Math.round(ms)}ms`;
            };

            const handleCopyJSON = async () => {
                try {
                    const jsonStr = JSON.stringify(event, null, 2);
                    await navigator.clipboard.writeText(jsonStr);
                    setCopySuccess(true);
                    setTimeout(() => setCopySuccess(false), 2000);
                } catch (err) {
                    console.error('Failed to copy:', err);
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = JSON.stringify(event, null, 2);
                    document.body.appendChild(textArea);
                    textArea.select();
                    try {
                        document.execCommand('copy');
                        setCopySuccess(true);
                        setTimeout(() => setCopySuccess(false), 2000);
                    } catch (e) {
                        console.error('Fallback copy failed:', e);
                    }
                    document.body.removeChild(textArea);
                }
            };

            // Render type-specific details
            const renderTypeSpecificDetails = () => {
                switch (event.event_type) {
                    case 'tool_call':
                        return <ToolCallDetails data={event.data} />;
                    case 'llm_request':
                        return <LLMRequestDetails data={event.data} />;
                    case 'error':
                        return <ErrorDetails data={event.data} />;
                    case 'reasoning':
                        return <ReasoningDetails data={event.data} />;
                    case 'progress':
                        return <ProgressDetails data={event.data} />;
                    default:
                        return (
                            <div>
                                <h4 className="text-sm font-semibold text-gray-500 mb-2">Event Data</h4>
                                <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 overflow-x-auto">
                                    <pre className="text-xs font-mono whitespace-pre-wrap text-gray-700">
                                        {JSON.stringify(event.data, null, 2)}
                                    </pre>
                                </div>
                            </div>
                        );
                }
            };

            return (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[85vh] flex flex-col">
                        {/* Header */}
                        <div className={`p-4 border-b ${colors.bg} rounded-t-lg flex items-center justify-between`}>
                            <div className="flex items-center gap-3">
                                <span className={`w-10 h-10 rounded-full ${colors.border.replace('border', 'bg').replace('400', '200')} ${colors.text} flex items-center justify-center text-xl font-bold`}>
                                    {colors.icon}
                                </span>
                                <div>
                                    <h3 className={`font-bold text-lg ${colors.text}`}>
                                        {event.event_type.replace('_', ' ').toUpperCase()}
                                    </h3>
                                    <p className="text-sm text-gray-500">Event #{event.sequence + 1}</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                {/* Copy as JSON button */}
                                <button
                                    onClick={handleCopyJSON}
                                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-all ${
                                        copySuccess
                                            ? 'bg-green-100 text-green-700'
                                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                    }`}
                                    title="Copy event data as JSON"
                                >
                                    {copySuccess ? (
                                        <>
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                            </svg>
                                            Copied!
                                        </>
                                    ) : (
                                        <>
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                            </svg>
                                            Copy JSON
                                        </>
                                    )}
                                </button>
                                {/* Close button */}
                                <button
                                    onClick={onClose}
                                    className="p-2 hover:bg-gray-200 rounded-full transition-colors"
                                    title="Close"
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                            </div>
                        </div>

                        {/* Content */}
                        <div className="p-4 overflow-y-auto flex-1">
                            {/* Summary */}
                            <div className="mb-4">
                                <h4 className="text-sm font-semibold text-gray-500 mb-1">Summary</h4>
                                <p className="text-gray-700 bg-gray-50 p-2 rounded">{event.summary}</p>
                            </div>

                            {/* Timing */}
                            <div className="mb-4 grid grid-cols-2 md:grid-cols-4 gap-3">
                                <div className="bg-gray-50 border border-gray-200 rounded-lg p-2">
                                    <div className="text-xs text-gray-500 font-semibold mb-1">Start Time</div>
                                    <div className="text-sm font-mono text-gray-700">{formatTime(event.relative_time_ms)}</div>
                                </div>
                                {event.duration_ms && (
                                    <div className="bg-gray-50 border border-gray-200 rounded-lg p-2">
                                        <div className="text-xs text-gray-500 font-semibold mb-1">Duration</div>
                                        <div className="text-sm font-mono text-gray-700">{formatTime(event.duration_ms)}</div>
                                    </div>
                                )}
                                {event.timestamp && (
                                    <div className="bg-gray-50 border border-gray-200 rounded-lg p-2 md:col-span-2">
                                        <div className="text-xs text-gray-500 font-semibold mb-1">Timestamp</div>
                                        <div className="text-xs font-mono text-gray-700">{new Date(event.timestamp).toLocaleString()}</div>
                                    </div>
                                )}
                            </div>

                            {/* Type-specific details */}
                            <div className="mb-4">
                                {renderTypeSpecificDetails()}
                            </div>

                            {/* Raw JSON (collapsible) */}
                            <details className="mt-4">
                                <summary className="cursor-pointer text-sm font-semibold text-gray-500 hover:text-gray-700 mb-2">
                                    Raw Event Data
                                </summary>
                                <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                                    <pre className="text-xs font-mono whitespace-pre-wrap text-gray-300">
                                        {JSON.stringify(event, null, 2)}
                                    </pre>
                                </div>
                            </details>
                        </div>
                    </div>
                </div>
            );
        }

        // TimelineContainer component - main container with zoom controls and filtering
        function TimelineContainer({ suiteName, testId, agents, availableAgents, onBack, tests }) {
            const [loading, setLoading] = useState(false);
            const [error, setError] = useState(null);
            const [timelineData, setTimelineData] = useState(null);
            const [selectedAgents, setSelectedAgents] = useState([]);
            const [selectedTestId, setSelectedTestId] = useState(testId || '');
            const [zoomLevel, setZoomLevel] = useState(1);
            const [eventFilter, setEventFilter] = useState('all');
            const [selectedEvent, setSelectedEvent] = useState(null);

            const loadTimeline = useCallback(async () => {
                if (selectedAgents.length < 1 || !selectedTestId) {
                    return;
                }

                setLoading(true);
                setError(null);

                try {
                    if (selectedAgents.length === 1) {
                        // Single agent timeline
                        const data = await api.get(
                            `/timeline/events?suite_name=${encodeURIComponent(suiteName)}&test_id=${encodeURIComponent(selectedTestId)}&agent_name=${encodeURIComponent(selectedAgents[0])}`
                        );
                        // Convert single timeline response to multi-timeline format
                        setTimelineData({
                            suite_name: data.suite_name,
                            test_id: data.test_id,
                            test_name: data.test_name,
                            timelines: [{
                                agent_name: data.agent_name,
                                test_execution_id: data.execution_id,
                                start_time: data.events[0]?.timestamp,
                                total_duration_ms: data.total_duration_ms || 0,
                                events: data.events,
                            }],
                        });
                    } else {
                        // Multi-agent timeline
                        const agentParams = selectedAgents.map(a => `agents=${encodeURIComponent(a)}`).join('&');
                        const data = await api.get(
                            `/timeline/compare?suite_name=${encodeURIComponent(suiteName)}&test_id=${encodeURIComponent(selectedTestId)}&${agentParams}`
                        );
                        setTimelineData(data);
                    }
                } catch (err) {
                    setError(err.message || 'Failed to load timeline data');
                    setTimelineData(null);
                } finally {
                    setLoading(false);
                }
            }, [suiteName, selectedTestId, selectedAgents]);

            // Calculate max duration for consistent scale
            const maxDuration = timelineData
                ? Math.max(...timelineData.timelines.map(t => t.total_duration_ms), 1)
                : 0;

            // Zoom handlers
            const handleZoomIn = () => setZoomLevel(prev => Math.min(prev * 1.5, 10));
            const handleZoomOut = () => setZoomLevel(prev => Math.max(prev / 1.5, 0.5));
            const handleZoomReset = () => setZoomLevel(1);

            // Event type counts (including total for 'all')
            const eventCounts = timelineData
                ? timelineData.timelines.reduce((acc, timeline) => {
                    timeline.events.forEach(e => {
                        acc[e.event_type] = (acc[e.event_type] || 0) + 1;
                        acc.all = (acc.all || 0) + 1;
                    });
                    return acc;
                }, { all: 0 })
                : { all: 0 };

            // Loading state
            if (loading) {
                return <SkeletonTimeline rows={selectedAgents.length || 2} />;
            }

            // Selection UI
            if (!timelineData) {
                return (
                    <div className="bg-white p-6 rounded-lg shadow">
                        {onBack && (
                            <button
                                onClick={onBack}
                                className="mb-4 text-blue-500 hover:text-blue-700 flex items-center gap-1"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                                </svg>
                                Back
                            </button>
                        )}
                        <h3 className="text-lg font-bold mb-4">Event Timeline</h3>

                        {error && (
                            <ErrorDisplay
                                error={{ message: error }}
                                onRetry={loadTimeline}
                                title="Failed to load timeline"
                            />
                        )}

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Select Test:
                                </label>
                                <select
                                    value={selectedTestId}
                                    onChange={(e) => setSelectedTestId(e.target.value)}
                                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                >
                                    <option value="">Choose a test...</option>
                                    {tests && tests.map((test) => (
                                        <option key={test.test_id} value={test.test_id}>
                                            {test.test_name || test.test_id}
                                        </option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Select Agents (1-3):
                                </label>
                                <AgentSelector
                                    agents={availableAgents}
                                    selectedAgents={selectedAgents}
                                    onSelectionChange={setSelectedAgents}
                                    maxAgents={3}
                                />
                            </div>
                        </div>

                        <button
                            onClick={loadTimeline}
                            disabled={selectedAgents.length < 1 || !selectedTestId}
                            className="w-full md:w-auto bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                        >
                            View Timeline
                        </button>

                        <p className="mt-4 text-sm text-gray-500">
                            Select a test and 1-3 agents to view their event timeline. Events are displayed chronologically with color-coding by type.
                        </p>
                    </div>
                );
            }

            // Timeline visualization
            return (
                <div>
                    {/* Header with test info and controls */}
                    <div className="bg-white p-4 rounded-lg shadow mb-4">
                        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                            <div>
                                <h3 className="text-lg font-bold">{timelineData.test_name}</h3>
                                <p className="text-sm text-gray-500">
                                    Suite: {timelineData.suite_name} | Test ID: {timelineData.test_id}
                                </p>
                            </div>
                            <button
                                onClick={() => setTimelineData(null)}
                                className="text-blue-500 hover:text-blue-700 flex items-center gap-1"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                                </svg>
                                Back to selection
                            </button>
                        </div>

                        {/* Zoom controls */}
                        <div className="flex flex-wrap items-center gap-4">
                            <div className="flex items-center gap-2">
                                <span className="text-sm text-gray-600">Zoom:</span>
                                <button
                                    onClick={handleZoomOut}
                                    className="p-1 rounded border border-gray-300 hover:bg-gray-100"
                                    title="Zoom out"
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                                    </svg>
                                </button>
                                <span className="text-sm font-mono w-12 text-center">{zoomLevel.toFixed(1)}x</span>
                                <button
                                    onClick={handleZoomIn}
                                    className="p-1 rounded border border-gray-300 hover:bg-gray-100"
                                    title="Zoom in"
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                                    </svg>
                                </button>
                                <button
                                    onClick={handleZoomReset}
                                    className="px-2 py-1 text-sm rounded border border-gray-300 hover:bg-gray-100"
                                    title="Reset zoom"
                                >
                                    Reset
                                </button>
                            </div>

                            {/* Event type filters */}
                            <EventFilters
                                eventFilter={eventFilter}
                                onFilterChange={setEventFilter}
                                eventCounts={eventCounts}
                                showCounts={true}
                            />
                        </div>
                    </div>

                    {/* Timeline visualization */}
                    <div className="bg-white p-4 rounded-lg shadow">
                        {/* Time scale */}
                        <TimeScale
                            totalDurationMs={maxDuration}
                            zoomLevel={zoomLevel}
                            width={800}
                        />

                        {/* Timeline rows */}
                        <div className="mt-4 overflow-x-auto pb-6">
                            <div style={{ width: `${100 * zoomLevel}%`, minWidth: '100%' }}>
                                {timelineData.timelines.map((timeline, idx) => (
                                    <TimelineRow
                                        key={timeline.agent_name}
                                        timeline={timeline}
                                        totalDurationMs={maxDuration}
                                        zoomLevel={zoomLevel}
                                        eventFilter={eventFilter}
                                        onEventClick={setSelectedEvent}
                                        rowIndex={idx}
                                    />
                                ))}
                            </div>
                        </div>

                        {/* Legend */}
                        <div className="mt-4 pt-4 border-t flex flex-wrap gap-4 text-xs">
                            <span className="font-semibold text-gray-600">Event Types:</span>
                            {Object.entries(EVENT_COLORS).map(([type, colors]) => (
                                <div key={type} className="flex items-center gap-1">
                                    <span className={`w-4 h-4 rounded ${colors.border.replace('border', 'bg').replace('400', '500')}`}></span>
                                    <span className={colors.text}>{type.replace('_', ' ')}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Event detail panel */}
                    {selectedEvent && (
                        <EventDetailPanel
                            event={selectedEvent}
                            onClose={() => setSelectedEvent(null)}
                        />
                    )}
                </div>
            );
        }

        // TimelineView component - wrapper with data loading (similar to LeaderboardView)
        function TimelineView({ suiteName, onBack }) {
            const [agents, setAgents] = useState([]);
            const [tests, setTests] = useState([]);
            const [loading, setLoading] = useState(true);

            useEffect(() => {
                Promise.all([
                    api.get('/agents'),
                    suiteName ? api.get(`/suites?suite_name=${encodeURIComponent(suiteName)}&limit=1`) : Promise.resolve({ items: [] }),
                ])
                    .then(async ([agentsData, suitesData]) => {
                        setAgents(agentsData);
                        if (suitesData.items && suitesData.items.length > 0) {
                            const detail = await api.get(`/suites/${suitesData.items[0].id}`);
                            if (detail && detail.tests) {
                                setTests(detail.tests);
                            }
                        }
                    })
                    .catch(console.error)
                    .finally(() => setLoading(false));
            }, [suiteName]);

            if (loading) {
                return (
                    <div className="bg-white p-6 rounded-lg shadow">
                        <SkeletonBox className="h-6 w-40 mb-4" />
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                            <div>
                                <SkeletonBox className="h-4 w-24 mb-2" />
                                <SkeletonBox className="h-10 w-full rounded" />
                            </div>
                            <div>
                                <SkeletonBox className="h-4 w-32 mb-2" />
                                <SkeletonBox className="h-10 w-full rounded" />
                            </div>
                        </div>
                        <SkeletonBox className="h-10 w-32 rounded" />
                    </div>
                );
            }

            return (
                <ErrorBoundary title="Timeline Error" message="Unable to display the timeline view.">
                    <TimelineContainer
                        suiteName={suiteName}
                        testId=""
                        agents={[]}
                        availableAgents={agents}
                        onBack={onBack}
                        tests={tests}
                    />
                </ErrorBoundary>
            );
        }

        // ==================== Leaderboard Matrix Components ====================

        // Score color coding based on score value
        const SCORE_COLORS = {
            excellent: { bg: 'bg-green-100', text: 'text-green-800', border: 'border-green-300' },
            good: { bg: 'bg-green-50', text: 'text-green-700', border: 'border-green-200' },
            medium: { bg: 'bg-yellow-50', text: 'text-yellow-700', border: 'border-yellow-200' },
            poor: { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200' },
            none: { bg: 'bg-gray-50', text: 'text-gray-400', border: 'border-gray-200' },
        };

        // Get color based on score value
        function getScoreColor(score) {
            if (score === null || score === undefined) return SCORE_COLORS.none;
            if (score >= 80) return SCORE_COLORS.excellent;
            if (score >= 60) return SCORE_COLORS.good;
            if (score >= 40) return SCORE_COLORS.medium;
            return SCORE_COLORS.poor;
        }

        // Difficulty badge colors
        const DIFFICULTY_COLORS = {
            easy: { bg: 'bg-green-100', text: 'text-green-700' },
            medium: { bg: 'bg-yellow-100', text: 'text-yellow-700' },
            hard: { bg: 'bg-orange-100', text: 'text-orange-700' },
            very_hard: { bg: 'bg-red-100', text: 'text-red-700' },
            unknown: { bg: 'bg-gray-100', text: 'text-gray-500' },
        };

        // ScoreCell component - displays a single score with color coding
        function ScoreCell({ testScore, agentName }) {
            if (!testScore) {
                return (
                    <td className="px-3 py-2 text-center">
                        <span className="text-gray-400">-</span>
                    </td>
                );
            }

            const { score, success, execution_count } = testScore;
            const colors = getScoreColor(score);

            return (
                <td className={`px-3 py-2 text-center ${colors.bg}`}>
                    <div className="flex flex-col items-center">
                        <span className={`font-semibold ${colors.text}`}>
                            {score !== null ? score.toFixed(0) : '-'}
                        </span>
                        <span className={`text-xs ${success ? 'text-green-600' : 'text-red-600'}`}>
                            {success ? 'Pass' : 'Fail'}
                        </span>
                        {execution_count > 1 && (
                            <span className="text-xs text-gray-400">
                                ({execution_count} runs)
                            </span>
                        )}
                    </div>
                </td>
            );
        }

        // AgentHeader component - column header showing agent stats
        function AgentHeader({ agent, onSort, sortConfig }) {
            const isActive = sortConfig.key === agent.agent_name;
            const rankBadge = agent.rank <= 3 ? ['bg-yellow-400', 'bg-gray-300', 'bg-amber-600'][agent.rank - 1] : 'bg-gray-200';

            return (
                <th
                    className="px-3 py-3 text-center bg-gray-50 cursor-pointer hover:bg-gray-100 transition-colors min-w-[120px]"
                    onClick={() => onSort(agent.agent_name)}
                >
                    <div className="flex flex-col items-center gap-1">
                        <div className="flex items-center gap-2">
                            <span className={`w-6 h-6 rounded-full ${rankBadge} flex items-center justify-center text-xs font-bold text-white`}>
                                {agent.rank}
                            </span>
                            <span className="font-semibold text-gray-700 truncate max-w-[100px]" title={agent.agent_name}>
                                {agent.agent_name}
                            </span>
                            {isActive && (
                                <svg className={`w-4 h-4 text-blue-500 ${sortConfig.direction === 'asc' ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                </svg>
                            )}
                        </div>
                        <div className="text-xs text-gray-500 space-y-0.5">
                            <div>Avg: {agent.avg_score !== null ? agent.avg_score.toFixed(1) : '-'}</div>
                            <div>Pass: {(agent.pass_rate * 100).toFixed(0)}%</div>
                            {agent.total_cost !== null && (
                                <div>${agent.total_cost.toFixed(2)}</div>
                            )}
                        </div>
                    </div>
                </th>
            );
        }

        // TestRow component - row showing test info and scores per agent
        function TestRow({ test, agents, rowIndex }) {
            const difficultyColors = DIFFICULTY_COLORS[test.difficulty] || DIFFICULTY_COLORS.unknown;

            return (
                <tr className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-3 py-2 sticky left-0 bg-inherit border-r border-gray-200 min-w-[200px]">
                        <div className="flex flex-col gap-1">
                            <span className="font-medium text-gray-800 truncate" title={test.test_name}>
                                {test.test_name}
                            </span>
                            <div className="flex flex-wrap gap-1">
                                <span className={`px-2 py-0.5 text-xs rounded ${difficultyColors.bg} ${difficultyColors.text}`}>
                                    {test.difficulty.replace('_', ' ')}
                                </span>
                                {test.pattern && (
                                    <span className="px-2 py-0.5 text-xs rounded bg-purple-100 text-purple-700">
                                        {test.pattern.replace('_', ' ')}
                                    </span>
                                )}
                            </div>
                            {test.tags && test.tags.length > 0 && (
                                <div className="flex flex-wrap gap-1">
                                    {test.tags.slice(0, 3).map((tag, idx) => (
                                        <span key={idx} className="px-1.5 py-0.5 text-xs rounded bg-blue-50 text-blue-600">
                                            {tag}
                                        </span>
                                    ))}
                                    {test.tags.length > 3 && (
                                        <span className="text-xs text-gray-400">+{test.tags.length - 3}</span>
                                    )}
                                </div>
                            )}
                        </div>
                    </td>
                    <td className="px-3 py-2 text-center border-r border-gray-200">
                        <span className={`font-semibold ${getScoreColor(test.avg_score).text}`}>
                            {test.avg_score !== null ? test.avg_score.toFixed(1) : '-'}
                        </span>
                    </td>
                    {agents.map((agent) => (
                        <ScoreCell
                            key={agent.agent_name}
                            testScore={test.scores_by_agent[agent.agent_name]}
                            agentName={agent.agent_name}
                        />
                    ))}
                </tr>
            );
        }

        // Pattern colors for test patterns
        const PATTERN_COLORS = {
            hard_for_all: { bg: 'bg-red-100', text: 'text-red-700' },
            easy: { bg: 'bg-green-100', text: 'text-green-700' },
            high_variance: { bg: 'bg-orange-100', text: 'text-orange-700' },
        };

        // Rank badge colors (1st, 2nd, 3rd)
        const RANK_BADGES = {
            1: { bg: 'bg-yellow-400', label: '1st' },
            2: { bg: 'bg-gray-300', label: '2nd' },
            3: { bg: 'bg-amber-600', label: '3rd' },
        };

        // AggregationRow component - summary row showing per-agent aggregations
        function AggregationRow({ tests, agents }) {
            // Calculate aggregated statistics
            const aggregations = useMemo(() => {
                // Per-test pattern counts
                const patternCounts = {
                    hard_for_all: 0,
                    easy: 0,
                    high_variance: 0,
                };
                tests.forEach((test) => {
                    if (test.pattern && patternCounts[test.pattern] !== undefined) {
                        patternCounts[test.pattern]++;
                    }
                });

                // Difficulty distribution
                const difficultyCounts = {
                    easy: 0,
                    medium: 0,
                    hard: 0,
                    very_hard: 0,
                    unknown: 0,
                };
                tests.forEach((test) => {
                    if (difficultyCounts[test.difficulty] !== undefined) {
                        difficultyCounts[test.difficulty]++;
                    }
                });

                // Overall average score across all tests
                const validAvgScores = tests
                    .map((t) => t.avg_score)
                    .filter((s) => s !== null && s !== undefined);
                const overallAvg = validAvgScores.length > 0
                    ? validAvgScores.reduce((a, b) => a + b, 0) / validAvgScores.length
                    : null;

                return {
                    patternCounts,
                    difficultyCounts,
                    overallAvg,
                };
            }, [tests]);

            return (
                <tr className="bg-blue-50 border-t-2 border-blue-200 font-semibold">
                    <td className="px-3 py-3 sticky left-0 bg-blue-50 border-r border-gray-200 min-w-[200px]">
                        <div className="flex flex-col gap-2">
                            <span className="font-bold text-gray-800">Summary</span>
                            <div className="flex flex-wrap gap-1 text-xs">
                                {aggregations.patternCounts.hard_for_all > 0 && (
                                    <span className={`px-2 py-0.5 rounded ${PATTERN_COLORS.hard_for_all.bg} ${PATTERN_COLORS.hard_for_all.text}`}>
                                        {aggregations.patternCounts.hard_for_all} hard for all
                                    </span>
                                )}
                                {aggregations.patternCounts.easy > 0 && (
                                    <span className={`px-2 py-0.5 rounded ${PATTERN_COLORS.easy.bg} ${PATTERN_COLORS.easy.text}`}>
                                        {aggregations.patternCounts.easy} easy
                                    </span>
                                )}
                                {aggregations.patternCounts.high_variance > 0 && (
                                    <span className={`px-2 py-0.5 rounded ${PATTERN_COLORS.high_variance.bg} ${PATTERN_COLORS.high_variance.text}`}>
                                        {aggregations.patternCounts.high_variance} high variance
                                    </span>
                                )}
                            </div>
                            <div className="flex flex-wrap gap-1 text-xs text-gray-500">
                                {Object.entries(aggregations.difficultyCounts)
                                    .filter(([_, count]) => count > 0)
                                    .map(([level, count]) => (
                                        <span key={level} className={`px-1.5 py-0.5 rounded ${(DIFFICULTY_COLORS[level] || DIFFICULTY_COLORS.unknown).bg} ${(DIFFICULTY_COLORS[level] || DIFFICULTY_COLORS.unknown).text}`}>
                                            {count} {level.replace('_', ' ')}
                                        </span>
                                    ))}
                            </div>
                        </div>
                    </td>
                    <td className="px-3 py-3 text-center border-r border-gray-200">
                        <span className={`font-bold ${getScoreColor(aggregations.overallAvg).text}`}>
                            {aggregations.overallAvg !== null ? aggregations.overallAvg.toFixed(1) : '-'}
                        </span>
                    </td>
                    {agents.map((agent) => {
                        const rankBadge = RANK_BADGES[agent.rank];
                        return (
                            <td key={agent.agent_name} className="px-3 py-3 text-center bg-blue-50">
                                <div className="flex flex-col items-center gap-1">
                                    {/* Rank badge */}
                                    {rankBadge && (
                                        <span className={`px-2 py-0.5 text-xs rounded-full ${rankBadge.bg} text-white font-bold`}>
                                            {rankBadge.label}
                                        </span>
                                    )}
                                    {/* Avg score */}
                                    <div className={`font-bold ${getScoreColor(agent.avg_score).text}`}>
                                        Avg: {agent.avg_score !== null ? agent.avg_score.toFixed(1) : '-'}
                                    </div>
                                    {/* Pass rate */}
                                    <div className={`text-xs ${agent.pass_rate >= 0.8 ? 'text-green-600' : agent.pass_rate >= 0.5 ? 'text-yellow-600' : 'text-red-600'}`}>
                                        Pass: {(agent.pass_rate * 100).toFixed(0)}%
                                    </div>
                                    {/* Tokens */}
                                    <div className="text-xs text-gray-500">
                                        {agent.total_tokens.toLocaleString()} tokens
                                    </div>
                                    {/* Cost */}
                                    {agent.total_cost !== null && (
                                        <div className="text-xs text-gray-500">
                                            ${agent.total_cost.toFixed(2)}
                                        </div>
                                    )}
                                </div>
                            </td>
                        );
                    })}
                </tr>
            );
        }

        // MatrixGrid component - main table showing tests vs agents
        function MatrixGrid({ data, onRefresh }) {
            const [sortConfig, setSortConfig] = useState({ key: null, direction: 'desc' });

            if (!data || !data.tests || data.tests.length === 0) {
                return (
                    <div className="bg-white p-8 rounded-lg shadow text-center">
                        <p className="text-gray-500">No test data available for this suite.</p>
                        <button
                            onClick={onRefresh}
                            className="mt-4 text-blue-500 hover:text-blue-700"
                        >
                            Refresh Data
                        </button>
                    </div>
                );
            }

            const handleSort = (key) => {
                setSortConfig((prev) => ({
                    key,
                    direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc',
                }));
            };

            // Sort tests based on sortConfig
            const sortedTests = [...data.tests].sort((a, b) => {
                if (!sortConfig.key) {
                    // Default sort by test_name
                    return a.test_name.localeCompare(b.test_name);
                }

                if (sortConfig.key === 'test_name') {
                    const cmp = a.test_name.localeCompare(b.test_name);
                    return sortConfig.direction === 'asc' ? cmp : -cmp;
                }

                if (sortConfig.key === 'avg_score') {
                    const aVal = a.avg_score ?? -1;
                    const bVal = b.avg_score ?? -1;
                    const cmp = aVal - bVal;
                    return sortConfig.direction === 'asc' ? cmp : -cmp;
                }

                if (sortConfig.key === 'difficulty') {
                    const order = { easy: 1, medium: 2, hard: 3, very_hard: 4, unknown: 5 };
                    const cmp = (order[a.difficulty] || 5) - (order[b.difficulty] || 5);
                    return sortConfig.direction === 'asc' ? cmp : -cmp;
                }

                // Sort by agent score
                const aScore = a.scores_by_agent[sortConfig.key]?.score ?? -1;
                const bScore = b.scores_by_agent[sortConfig.key]?.score ?? -1;
                const cmp = aScore - bScore;
                return sortConfig.direction === 'asc' ? cmp : -cmp;
            });

            return (
                <div className="bg-white rounded-lg shadow">
                    {/* Matrix header with summary */}
                    <div className="p-4 border-b flex flex-wrap items-center justify-between gap-4">
                        <div>
                            <h3 className="text-lg font-bold text-gray-800">Leaderboard Matrix</h3>
                            <p className="text-sm text-gray-500">
                                {data.total_tests} tests × {data.total_agents} agents
                            </p>
                        </div>
                        <div className="flex gap-2">
                            <button
                                onClick={() => handleSort('test_name')}
                                className={`px-3 py-1 text-sm rounded ${sortConfig.key === 'test_name' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                            >
                                Sort by Name
                            </button>
                            <button
                                onClick={() => handleSort('avg_score')}
                                className={`px-3 py-1 text-sm rounded ${sortConfig.key === 'avg_score' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                            >
                                Sort by Avg Score
                            </button>
                            <button
                                onClick={() => handleSort('difficulty')}
                                className={`px-3 py-1 text-sm rounded ${sortConfig.key === 'difficulty' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                            >
                                Sort by Difficulty
                            </button>
                        </div>
                    </div>

                    {/* Scrollable table container */}
                    <div className="overflow-x-auto matrix-scroll-container">
                        <table className="min-w-full border-collapse">
                            <thead>
                                <tr>
                                    <th className="px-3 py-3 text-left bg-gray-100 sticky left-0 z-10 border-r border-gray-200 min-w-[200px]">
                                        <span className="font-semibold text-gray-700">Test</span>
                                    </th>
                                    <th
                                        className="px-3 py-3 text-center bg-gray-100 cursor-pointer hover:bg-gray-200 min-w-[80px] border-r border-gray-200"
                                        onClick={() => handleSort('avg_score')}
                                    >
                                        <div className="flex items-center justify-center gap-1">
                                            <span className="font-semibold text-gray-700">Avg</span>
                                            {sortConfig.key === 'avg_score' && (
                                                <svg className={`w-4 h-4 text-blue-500 ${sortConfig.direction === 'asc' ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                                </svg>
                                            )}
                                        </div>
                                    </th>
                                    {data.agents.map((agent) => (
                                        <AgentHeader
                                            key={agent.agent_name}
                                            agent={agent}
                                            onSort={handleSort}
                                            sortConfig={sortConfig}
                                        />
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {sortedTests.map((test, idx) => (
                                    <TestRow
                                        key={test.test_id}
                                        test={test}
                                        agents={data.agents}
                                        rowIndex={idx}
                                    />
                                ))}
                            </tbody>
                            <tfoot>
                                <AggregationRow
                                    tests={data.tests}
                                    agents={data.agents}
                                />
                            </tfoot>
                        </table>
                    </div>

                    {/* Legend */}
                    <div className="p-4 border-t bg-gray-50 flex flex-wrap gap-4 text-xs">
                        <div className="flex items-center gap-2">
                            <span className="font-semibold text-gray-600">Score:</span>
                            <span className="px-2 py-0.5 rounded bg-green-100 text-green-800">80+</span>
                            <span className="px-2 py-0.5 rounded bg-green-50 text-green-700">60-79</span>
                            <span className="px-2 py-0.5 rounded bg-yellow-50 text-yellow-700">40-59</span>
                            <span className="px-2 py-0.5 rounded bg-red-50 text-red-700">&lt;40</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="font-semibold text-gray-600">Rank:</span>
                            <span className="w-5 h-5 rounded-full bg-yellow-400 flex items-center justify-center text-white text-xs font-bold">1</span>
                            <span className="w-5 h-5 rounded-full bg-gray-300 flex items-center justify-center text-white text-xs font-bold">2</span>
                            <span className="w-5 h-5 rounded-full bg-amber-600 flex items-center justify-center text-white text-xs font-bold">3</span>
                        </div>
                    </div>
                </div>
            );
        }

        // Leaderboard view component - wrapper with data loading
        function LeaderboardView({ suiteName, onBack }) {
            const [matrixData, setMatrixData] = useState(null);
            const [loading, setLoading] = useState(true);
            const [error, setError] = useState(null);

            const loadMatrix = useCallback(async () => {
                if (!suiteName) return;
                setLoading(true);
                setError(null);
                try {
                    const data = await api.get(`/leaderboard/matrix?suite_name=${encodeURIComponent(suiteName)}`);
                    setMatrixData(data);
                } catch (err) {
                    setError(err.message || 'Failed to load leaderboard data');
                    setMatrixData(null);
                } finally {
                    setLoading(false);
                }
            }, [suiteName]);

            useEffect(() => {
                loadMatrix();
            }, [loadMatrix]);

            if (loading) {
                return <SkeletonLeaderboardMatrix rows={6} cols={3} />;
            }

            if (error) {
                return (
                    <ErrorDisplay
                        error={{ message: error }}
                        onRetry={loadMatrix}
                        title="Error loading leaderboard"
                    />
                );
            }

            return (
                <ErrorBoundary title="Leaderboard Error" onRetry={loadMatrix}>
                    <div>
                        {onBack && (
                            <button
                                onClick={onBack}
                                className="mb-4 text-blue-500 hover:text-blue-700 flex items-center gap-1"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                                </svg>
                                Back
                            </button>
                        )}
                        <MatrixGrid data={matrixData} onRefresh={loadMatrix} />
                    </div>
                </ErrorBoundary>
            );
        }

        // ==================== End Leaderboard Matrix Components ====================

        // Trend chart component
        function TrendChart({ suiteName }) {
            const chartRef = React.useRef(null);
            const chartInstance = React.useRef(null);

            useEffect(() => {
                if (!suiteName || !chartRef.current) return;

                api.get(`/trends/suite?suite_name=${encodeURIComponent(suiteName)}&metric=success_rate&limit=20`)
                    .then((data) => {
                        if (chartInstance.current) {
                            chartInstance.current.destroy();
                        }

                        const trend = data.suite_trends[0];
                        if (!trend || !trend.data_points.length) return;

                        const ctx = chartRef.current.getContext('2d');
                        chartInstance.current = new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: trend.data_points.map(p =>
                                    new Date(p.timestamp).toLocaleDateString()
                                ),
                                datasets: [{
                                    label: 'Success Rate',
                                    data: trend.data_points.map(p => p.value * 100),
                                    borderColor: 'rgb(59, 130, 246)',
                                    tension: 0.1,
                                    fill: false
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: {
                                    y: {
                                        beginAtZero: true,
                                        max: 100,
                                        title: { display: true, text: 'Success Rate (%)' }
                                    }
                                }
                            }
                        });
                    })
                    .catch(console.error);

                return () => {
                    if (chartInstance.current) {
                        chartInstance.current.destroy();
                    }
                };
            }, [suiteName]);

            return (
                <div className="bg-white p-4 rounded shadow">
                    <h3 className="font-bold mb-4">Historical Trends</h3>
                    <div className="chart-container">
                        <canvas ref={chartRef}></canvas>
                    </div>
                </div>
            );
        }

        // ==================== Test Creator Form Components ====================

        // Category badge colors for templates
        const CATEGORY_COLORS = {
            code: { bg: 'bg-blue-100', text: 'text-blue-700', border: 'border-blue-200' },
            file: { bg: 'bg-green-100', text: 'text-green-700', border: 'border-green-200' },
            data: { bg: 'bg-purple-100', text: 'text-purple-700', border: 'border-purple-200' },
            web: { bg: 'bg-orange-100', text: 'text-orange-700', border: 'border-orange-200' },
            api: { bg: 'bg-pink-100', text: 'text-pink-700', border: 'border-pink-200' },
            default: { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-200' },
        };

        // Get category color
        function getCategoryColor(category) {
            const normalizedCategory = category?.toLowerCase() || 'default';
            return CATEGORY_COLORS[normalizedCategory] || CATEGORY_COLORS.default;
        }

        // Skeleton for test creator
        function SkeletonTestCreator() {
            return (
                <div className="bg-white rounded-lg shadow p-6">
                    {/* Progress indicator skeleton */}
                    <div className="flex items-center justify-center gap-4 mb-8">
                        {[1, 2, 3].map((i) => (
                            <div key={i} className="flex items-center">
                                <SkeletonBox className="w-8 h-8 rounded-full" />
                                <SkeletonBox className="h-4 w-24 ml-2" />
                                {i < 3 && <SkeletonBox className="w-12 h-0.5 mx-4" />}
                            </div>
                        ))}
                    </div>
                    {/* Form skeleton */}
                    <div className="space-y-4">
                        <SkeletonBox className="h-10 w-full rounded" />
                        <SkeletonBox className="h-24 w-full rounded" />
                        <div className="grid grid-cols-2 gap-4">
                            <SkeletonBox className="h-10 w-full rounded" />
                            <SkeletonBox className="h-10 w-full rounded" />
                        </div>
                    </div>
                </div>
            );
        }

        // Step indicator component
        function StepIndicator({ currentStep, steps }) {
            return (
                <div className="flex items-center justify-center gap-2 mb-8">
                    {steps.map((step, index) => (
                        <React.Fragment key={step.id}>
                            <div className="flex items-center">
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center font-semibold text-sm transition-all ${
                                    currentStep === index
                                        ? 'bg-blue-600 text-white ring-4 ring-blue-100'
                                        : currentStep > index
                                            ? 'bg-green-500 text-white'
                                            : 'bg-gray-200 text-gray-500'
                                }`}>
                                    {currentStep > index ? (
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                        </svg>
                                    ) : (
                                        index + 1
                                    )}
                                </div>
                                <span className={`ml-2 text-sm font-medium hidden sm:inline ${
                                    currentStep === index ? 'text-blue-600' : 'text-gray-500'
                                }`}>
                                    {step.label}
                                </span>
                            </div>
                            {index < steps.length - 1 && (
                                <div className={`w-8 md:w-16 h-0.5 ${
                                    currentStep > index ? 'bg-green-500' : 'bg-gray-200'
                                }`} />
                            )}
                        </React.Fragment>
                    ))}
                </div>
            );
        }

        // Template card component
        function TemplateCard({ template, isSelected, onSelect }) {
            const colors = getCategoryColor(template.category);

            return (
                <div
                    onClick={() => onSelect(template)}
                    className={`cursor-pointer border-2 rounded-lg p-4 transition-all hover:shadow-md ${
                        isSelected
                            ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                            : 'border-gray-200 hover:border-gray-300'
                    }`}
                >
                    <div className="flex items-start justify-between mb-2">
                        <h4 className="font-semibold text-gray-800">{template.name}</h4>
                        <span className={`px-2 py-0.5 text-xs rounded-full ${colors.bg} ${colors.text}`}>
                            {template.category}
                        </span>
                    </div>
                    <p className="text-sm text-gray-600 mb-3 line-clamp-2">{template.description}</p>
                    {template.tags && template.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                            {template.tags.slice(0, 3).map((tag, idx) => (
                                <span key={idx} className="px-2 py-0.5 text-xs rounded bg-gray-100 text-gray-600">
                                    {tag}
                                </span>
                            ))}
                            {template.tags.length > 3 && (
                                <span className="text-xs text-gray-400">+{template.tags.length - 3}</span>
                            )}
                        </div>
                    )}
                    {template.variables && template.variables.length > 0 && (
                        <div className="mt-2 text-xs text-gray-500">
                            Variables: {template.variables.join(', ')}
                        </div>
                    )}
                </div>
            );
        }

        // Test item component for the test list
        function TestItem({ test, onRemove, onEdit }) {
            return (
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200 hover:bg-gray-100 transition-colors">
                    <div className="flex-grow min-w-0">
                        <div className="flex items-center gap-2">
                            <span className="font-medium text-gray-800 truncate">{test.name}</span>
                            <span className="text-xs text-gray-500 bg-gray-200 px-2 py-0.5 rounded">
                                {test.id}
                            </span>
                        </div>
                        {test.description && (
                            <p className="text-sm text-gray-600 truncate mt-1">{test.description}</p>
                        )}
                        {test.tags && test.tags.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-1">
                                {test.tags.slice(0, 3).map((tag, idx) => (
                                    <span key={idx} className="px-1.5 py-0.5 text-xs rounded bg-blue-50 text-blue-600">
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        )}
                    </div>
                    <div className="flex items-center gap-2 ml-4">
                        <button
                            onClick={() => onEdit(test)}
                            className="p-1.5 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded transition-colors"
                            title="Edit test"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                            </svg>
                        </button>
                        <button
                            onClick={() => onRemove(test.id)}
                            className="p-1.5 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                            title="Remove test"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                        </button>
                    </div>
                </div>
            );
        }

        // Step 1: Suite Details Form
        function SuiteDetailsStep({ suiteData, onChange, errors }) {
            const handleChange = (field, value) => {
                onChange({ ...suiteData, [field]: value });
            };

            const handleDefaultsChange = (field, value) => {
                onChange({
                    ...suiteData,
                    defaults: { ...suiteData.defaults, [field]: value }
                });
            };

            const handleScoringChange = (field, value) => {
                onChange({
                    ...suiteData,
                    defaults: {
                        ...suiteData.defaults,
                        scoring: { ...suiteData.defaults.scoring, [field]: value }
                    }
                });
            };

            return (
                <div className="space-y-6">
                    {/* Basic Info */}
                    <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-semibold text-gray-700 mb-4">Basic Information</h4>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Suite Name <span className="text-red-500">*</span>
                                </label>
                                <input
                                    type="text"
                                    value={suiteData.name}
                                    onChange={(e) => handleChange('name', e.target.value)}
                                    placeholder="e.g., My Agent Test Suite"
                                    className={`w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                                        errors.name ? 'border-red-500' : 'border-gray-300'
                                    }`}
                                />
                                {errors.name && (
                                    <p className="text-red-500 text-xs mt-1">{errors.name}</p>
                                )}
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Description
                                </label>
                                <textarea
                                    value={suiteData.description || ''}
                                    onChange={(e) => handleChange('description', e.target.value)}
                                    placeholder="Describe what this test suite evaluates..."
                                    rows={3}
                                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                />
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Version
                                    </label>
                                    <input
                                        type="text"
                                        value={suiteData.version}
                                        onChange={(e) => handleChange('version', e.target.value)}
                                        placeholder="1.0"
                                        className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Runs Per Test
                                    </label>
                                    <input
                                        type="number"
                                        value={suiteData.defaults.runs_per_test}
                                        onChange={(e) => handleDefaultsChange('runs_per_test', parseInt(e.target.value) || 1)}
                                        min={1}
                                        max={100}
                                        className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    />
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Default Constraints */}
                    <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-semibold text-gray-700 mb-4">Default Constraints</h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Timeout (seconds)
                                </label>
                                <input
                                    type="number"
                                    value={suiteData.defaults.timeout_seconds}
                                    onChange={(e) => handleDefaultsChange('timeout_seconds', parseInt(e.target.value) || 300)}
                                    min={1}
                                    max={3600}
                                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Max Steps (optional)
                                </label>
                                <input
                                    type="number"
                                    value={suiteData.defaults.constraints?.max_steps || ''}
                                    onChange={(e) => {
                                        const value = e.target.value ? parseInt(e.target.value) : null;
                                        onChange({
                                            ...suiteData,
                                            defaults: {
                                                ...suiteData.defaults,
                                                constraints: {
                                                    ...suiteData.defaults.constraints,
                                                    max_steps: value
                                                }
                                            }
                                        });
                                    }}
                                    min={1}
                                    placeholder="No limit"
                                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Scoring Weights */}
                    <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-semibold text-gray-700 mb-4">Default Scoring Weights</h4>
                        <div className="space-y-4">
                            {[
                                { key: 'quality_weight', label: 'Quality', value: suiteData.defaults.scoring.quality_weight },
                                { key: 'completeness_weight', label: 'Completeness', value: suiteData.defaults.scoring.completeness_weight },
                                { key: 'efficiency_weight', label: 'Efficiency', value: suiteData.defaults.scoring.efficiency_weight },
                                { key: 'cost_weight', label: 'Cost', value: suiteData.defaults.scoring.cost_weight },
                            ].map(({ key, label, value }) => (
                                <div key={key} className="flex items-center gap-4">
                                    <label className="w-28 text-sm font-medium text-gray-700">{label}</label>
                                    <input
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.1"
                                        value={value}
                                        onChange={(e) => handleScoringChange(key, parseFloat(e.target.value))}
                                        className="flex-grow h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                    />
                                    <span className="w-12 text-sm text-gray-600 text-right">{(value * 100).toFixed(0)}%</span>
                                </div>
                            ))}
                            <p className="text-xs text-gray-500">
                                Total: {(
                                    (suiteData.defaults.scoring.quality_weight +
                                    suiteData.defaults.scoring.completeness_weight +
                                    suiteData.defaults.scoring.efficiency_weight +
                                    suiteData.defaults.scoring.cost_weight) * 100
                                ).toFixed(0)}% (should equal 100%)
                            </p>
                        </div>
                    </div>
                </div>
            );
        }

        // Step 2: Template Selection and Test List
        function TemplateSelectionStep({ templates, tests, onAddTest, onRemoveTest, onEditTest, selectedCategory, onCategoryChange, loading }) {
            const [selectedTemplate, setSelectedTemplate] = useState(null);
            const [showAddModal, setShowAddModal] = useState(false);
            const [editingTest, setEditingTest] = useState(null);

            // Filter templates by category
            const filteredTemplates = selectedCategory === 'all'
                ? templates
                : templates.filter(t => t.category === selectedCategory);

            // Get unique categories
            const categories = ['all', ...new Set(templates.map(t => t.category))];

            const handleAddFromTemplate = () => {
                if (selectedTemplate) {
                    setEditingTest({
                        id: `test-${Date.now()}`,
                        name: selectedTemplate.name,
                        description: selectedTemplate.description,
                        tags: [...(selectedTemplate.tags || [])],
                        task: {
                            description: selectedTemplate.task_template,
                            input_data: null,
                            expected_artifacts: null
                        },
                        constraints: selectedTemplate.default_constraints,
                        assertions: selectedTemplate.default_assertions,
                        scoring: null
                    });
                    setShowAddModal(true);
                }
            };

            const handleAddCustom = () => {
                setSelectedTemplate(null);
                setEditingTest({
                    id: `test-${Date.now()}`,
                    name: '',
                    description: '',
                    tags: [],
                    task: {
                        description: '',
                        input_data: null,
                        expected_artifacts: null
                    },
                    constraints: {
                        max_steps: null,
                        max_tokens: null,
                        timeout_seconds: 300,
                        allowed_tools: null,
                        budget_usd: null
                    },
                    assertions: [],
                    scoring: null
                });
                setShowAddModal(true);
            };

            const handleSaveTest = (test) => {
                onAddTest(test);
                setShowAddModal(false);
                setEditingTest(null);
                setSelectedTemplate(null);
            };

            const handleEditExisting = (test) => {
                setEditingTest(test);
                setShowAddModal(true);
            };

            if (loading) {
                return (
                    <div className="space-y-4">
                        <SkeletonBox className="h-10 w-full rounded" />
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {[1, 2, 3, 4, 5, 6].map((i) => (
                                <div key={i} className="border rounded-lg p-4">
                                    <SkeletonBox className="h-5 w-3/4 mb-2" />
                                    <SkeletonBox className="h-4 w-full mb-1" />
                                    <SkeletonBox className="h-4 w-2/3" />
                                </div>
                            ))}
                        </div>
                    </div>
                );
            }

            return (
                <div className="space-y-6">
                    {/* Category Filter */}
                    <div className="flex flex-wrap gap-2 items-center">
                        <span className="text-sm font-medium text-gray-700">Category:</span>
                        {categories.map((cat) => {
                            const colors = cat === 'all'
                                ? { bg: 'bg-gray-100', text: 'text-gray-700' }
                                : getCategoryColor(cat);
                            return (
                                <button
                                    key={cat}
                                    onClick={() => onCategoryChange(cat)}
                                    className={`px-3 py-1.5 text-sm rounded-full transition-all ${
                                        selectedCategory === cat
                                            ? `${colors.bg} ${colors.text} ring-2 ring-offset-1 ring-blue-300`
                                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                    }`}
                                >
                                    {cat === 'all' ? 'All' : cat.charAt(0).toUpperCase() + cat.slice(1)}
                                </button>
                            );
                        })}
                    </div>

                    {/* Template Grid */}
                    <div>
                        <h4 className="font-semibold text-gray-700 mb-3">Available Templates</h4>
                        {filteredTemplates.length === 0 ? (
                            <div className="text-center py-8 text-gray-500 bg-gray-50 rounded-lg">
                                No templates found in this category.
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-80 overflow-y-auto p-1">
                                {filteredTemplates.map((template) => (
                                    <TemplateCard
                                        key={template.name}
                                        template={template}
                                        isSelected={selectedTemplate?.name === template.name}
                                        onSelect={setSelectedTemplate}
                                    />
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Add Test Buttons */}
                    <div className="flex flex-wrap gap-3">
                        <button
                            onClick={handleAddFromTemplate}
                            disabled={!selectedTemplate}
                            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                            </svg>
                            Add from Template
                        </button>
                        <button
                            onClick={handleAddCustom}
                            className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                            </svg>
                            Add Custom Test
                        </button>
                    </div>

                    {/* Test List */}
                    <div>
                        <div className="flex items-center justify-between mb-3">
                            <h4 className="font-semibold text-gray-700">Tests in Suite ({tests.length})</h4>
                        </div>
                        {tests.length === 0 ? (
                            <div className="text-center py-8 text-gray-500 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                                <svg className="w-12 h-12 mx-auto mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                                </svg>
                                <p>No tests added yet.</p>
                                <p className="text-sm mt-1">Select a template above or add a custom test.</p>
                            </div>
                        ) : (
                            <div className="space-y-2 max-h-64 overflow-y-auto">
                                {tests.map((test) => (
                                    <TestItem
                                        key={test.id}
                                        test={test}
                                        onRemove={onRemoveTest}
                                        onEdit={handleEditExisting}
                                    />
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Add/Edit Test Modal */}
                    {showAddModal && editingTest && (
                        <TestEditModal
                            test={editingTest}
                            onSave={handleSaveTest}
                            onCancel={() => {
                                setShowAddModal(false);
                                setEditingTest(null);
                            }}
                            isEditing={tests.some(t => t.id === editingTest.id)}
                        />
                    )}
                </div>
            );
        }

        // Test Edit Modal
        function TestEditModal({ test, onSave, onCancel, isEditing }) {
            const [formData, setFormData] = useState(test);
            const [errors, setErrors] = useState({});

            const handleChange = (field, value) => {
                setFormData({ ...formData, [field]: value });
                if (errors[field]) {
                    setErrors({ ...errors, [field]: null });
                }
            };

            const handleTaskChange = (field, value) => {
                setFormData({
                    ...formData,
                    task: { ...formData.task, [field]: value }
                });
            };

            const handleSubmit = () => {
                const newErrors = {};
                if (!formData.id.trim()) newErrors.id = 'Test ID is required';
                if (!formData.name.trim()) newErrors.name = 'Test name is required';
                if (!formData.task.description.trim()) newErrors.description = 'Task description is required';

                if (Object.keys(newErrors).length > 0) {
                    setErrors(newErrors);
                    return;
                }

                onSave(formData);
            };

            return (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                        <div className="p-6 border-b">
                            <h3 className="text-lg font-bold text-gray-800">
                                {isEditing ? 'Edit Test' : 'Add New Test'}
                            </h3>
                        </div>
                        <div className="p-6 space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Test ID <span className="text-red-500">*</span>
                                    </label>
                                    <input
                                        type="text"
                                        value={formData.id}
                                        onChange={(e) => handleChange('id', e.target.value)}
                                        placeholder="e.g., test-file-creation"
                                        disabled={isEditing}
                                        className={`w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                                            errors.id ? 'border-red-500' : 'border-gray-300'
                                        } ${isEditing ? 'bg-gray-100' : ''}`}
                                    />
                                    {errors.id && <p className="text-red-500 text-xs mt-1">{errors.id}</p>}
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Test Name <span className="text-red-500">*</span>
                                    </label>
                                    <input
                                        type="text"
                                        value={formData.name}
                                        onChange={(e) => handleChange('name', e.target.value)}
                                        placeholder="e.g., File Creation Test"
                                        className={`w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                                            errors.name ? 'border-red-500' : 'border-gray-300'
                                        }`}
                                    />
                                    {errors.name && <p className="text-red-500 text-xs mt-1">{errors.name}</p>}
                                </div>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Description
                                </label>
                                <input
                                    type="text"
                                    value={formData.description || ''}
                                    onChange={(e) => handleChange('description', e.target.value)}
                                    placeholder="Brief description of the test"
                                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Task Description <span className="text-red-500">*</span>
                                </label>
                                <textarea
                                    value={formData.task.description}
                                    onChange={(e) => handleTaskChange('description', e.target.value)}
                                    placeholder="Describe the task the agent should perform..."
                                    rows={4}
                                    className={`w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                                        errors.description ? 'border-red-500' : 'border-gray-300'
                                    }`}
                                />
                                {errors.description && <p className="text-red-500 text-xs mt-1">{errors.description}</p>}
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Tags (comma-separated)
                                </label>
                                <input
                                    type="text"
                                    value={(formData.tags || []).join(', ')}
                                    onChange={(e) => handleChange('tags', e.target.value.split(',').map(t => t.trim()).filter(t => t))}
                                    placeholder="e.g., file, basic, smoke"
                                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                />
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Timeout (seconds)
                                    </label>
                                    <input
                                        type="number"
                                        value={formData.constraints?.timeout_seconds || 300}
                                        onChange={(e) => handleChange('constraints', {
                                            ...formData.constraints,
                                            timeout_seconds: parseInt(e.target.value) || 300
                                        })}
                                        min={1}
                                        className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Max Steps (optional)
                                    </label>
                                    <input
                                        type="number"
                                        value={formData.constraints?.max_steps || ''}
                                        onChange={(e) => handleChange('constraints', {
                                            ...formData.constraints,
                                            max_steps: e.target.value ? parseInt(e.target.value) : null
                                        })}
                                        min={1}
                                        placeholder="No limit"
                                        className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    />
                                </div>
                            </div>
                        </div>
                        <div className="p-6 border-t bg-gray-50 flex justify-end gap-3">
                            <button
                                onClick={onCancel}
                                className="px-4 py-2 text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300 transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleSubmit}
                                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                            >
                                {isEditing ? 'Update Test' : 'Add Test'}
                            </button>
                        </div>
                    </div>
                </div>
            );
        }

        // Step 3: YAML Preview and Save
        function YAMLPreviewStep({ suiteData, tests, onSave, saving, error }) {
            const [yamlPreview, setYamlPreview] = useState('');
            const [copied, setCopied] = useState(false);

            // Generate YAML preview
            useEffect(() => {
                const yaml = generateYAMLPreview(suiteData, tests);
                setYamlPreview(yaml);
            }, [suiteData, tests]);

            const handleCopy = async () => {
                try {
                    await navigator.clipboard.writeText(yamlPreview);
                    setCopied(true);
                    setTimeout(() => setCopied(false), 2000);
                } catch (err) {
                    console.error('Failed to copy:', err);
                }
            };

            return (
                <div className="space-y-6">
                    {/* Summary */}
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <h4 className="font-semibold text-blue-800 mb-2">Suite Summary</h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                                <span className="text-blue-600">Name:</span>
                                <span className="ml-2 font-medium text-blue-900">{suiteData.name}</span>
                            </div>
                            <div>
                                <span className="text-blue-600">Version:</span>
                                <span className="ml-2 font-medium text-blue-900">{suiteData.version}</span>
                            </div>
                            <div>
                                <span className="text-blue-600">Tests:</span>
                                <span className="ml-2 font-medium text-blue-900">{tests.length}</span>
                            </div>
                            <div>
                                <span className="text-blue-600">Runs/Test:</span>
                                <span className="ml-2 font-medium text-blue-900">{suiteData.defaults.runs_per_test}</span>
                            </div>
                        </div>
                    </div>

                    {/* YAML Preview */}
                    <div>
                        <div className="flex items-center justify-between mb-2">
                            <h4 className="font-semibold text-gray-700">YAML Preview</h4>
                            <button
                                onClick={handleCopy}
                                className="flex items-center gap-2 px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
                            >
                                {copied ? (
                                    <>
                                        <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                        </svg>
                                        Copied!
                                    </>
                                ) : (
                                    <>
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                        </svg>
                                        Copy to Clipboard
                                    </>
                                )}
                            </button>
                        </div>
                        <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto max-h-96">
                            <pre className="text-sm text-gray-100 font-mono whitespace-pre">{yamlPreview}</pre>
                        </div>
                    </div>

                    {/* Error Message */}
                    {error && (
                        <ErrorDisplay
                            error={{ message: error }}
                            title="Failed to save suite"
                        />
                    )}

                    {/* Save Button */}
                    <div className="flex justify-end">
                        <button
                            onClick={onSave}
                            disabled={saving || tests.length === 0}
                            className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
                        >
                            {saving ? (
                                <>
                                    <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Saving...
                                </>
                            ) : (
                                <>
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
                                    </svg>
                                    Save Suite Definition
                                </>
                            )}
                        </button>
                    </div>

                    {tests.length === 0 && (
                        <p className="text-center text-amber-600 text-sm">
                            Please add at least one test before saving.
                        </p>
                    )}
                </div>
            );
        }

        // Helper function to generate YAML preview
        function generateYAMLPreview(suiteData, tests) {
            const indent = (str, spaces) => str.split('\n').map(line => ' '.repeat(spaces) + line).join('\n');

            let yaml = `# ATP Test Suite Definition
# Generated by ATP Dashboard

name: "${suiteData.name}"
version: "${suiteData.version}"
`;

            if (suiteData.description) {
                yaml += `description: "${suiteData.description}"

`;
            } else {
                yaml += `
`;
            }

            yaml += `defaults:
  runs_per_test: ${suiteData.defaults.runs_per_test}
  timeout_seconds: ${suiteData.defaults.timeout_seconds}
  scoring:
    quality_weight: ${suiteData.defaults.scoring.quality_weight}
    completeness_weight: ${suiteData.defaults.scoring.completeness_weight}
    efficiency_weight: ${suiteData.defaults.scoring.efficiency_weight}
    cost_weight: ${suiteData.defaults.scoring.cost_weight}

tests:
`;

            tests.forEach((test) => {
                yaml += `  - id: "${test.id}"
    name: "${test.name}"
`;
                if (test.description) {
                    yaml += `    description: "${test.description}"
`;
                }
                if (test.tags && test.tags.length > 0) {
                    yaml += `    tags: [${test.tags.map(t => `"${t}"`).join(', ')}]
`;
                }
                yaml += `    task:
      description: |
${indent(test.task.description, 8)}
    constraints:
      timeout_seconds: ${test.constraints?.timeout_seconds || 300}
`;
                if (test.constraints?.max_steps) {
                    yaml += `      max_steps: ${test.constraints.max_steps}
`;
                }
                yaml += `
`;
            });

            return yaml.trim();
        }

        // Main TestCreatorForm component
        function TestCreatorForm({ onClose, onSuccess }) {
            const [currentStep, setCurrentStep] = useState(0);
            const [suiteData, setSuiteData] = useState({
                name: '',
                version: '1.0',
                description: '',
                defaults: {
                    runs_per_test: 1,
                    timeout_seconds: 300,
                    scoring: {
                        quality_weight: 0.4,
                        completeness_weight: 0.3,
                        efficiency_weight: 0.2,
                        cost_weight: 0.1
                    },
                    constraints: null
                },
                agents: []
            });
            const [tests, setTests] = useState([]);
            const [templates, setTemplates] = useState([]);
            const [selectedCategory, setSelectedCategory] = useState('all');
            const [loading, setLoading] = useState(true);
            const [saving, setSaving] = useState(false);
            const [errors, setErrors] = useState({});
            const [saveError, setSaveError] = useState(null);

            const steps = [
                { id: 'details', label: 'Suite Details' },
                { id: 'tests', label: 'Tests' },
                { id: 'preview', label: 'Preview & Save' }
            ];

            // Load templates on mount
            useEffect(() => {
                api.get('/templates')
                    .then((data) => {
                        setTemplates(data.templates || []);
                    })
                    .catch(console.error)
                    .finally(() => setLoading(false));
            }, []);

            const validateStep = (step) => {
                const newErrors = {};

                if (step === 0) {
                    if (!suiteData.name.trim()) {
                        newErrors.name = 'Suite name is required';
                    }
                }

                setErrors(newErrors);
                return Object.keys(newErrors).length === 0;
            };

            const handleNext = () => {
                if (validateStep(currentStep)) {
                    setCurrentStep(Math.min(currentStep + 1, steps.length - 1));
                }
            };

            const handleBack = () => {
                setCurrentStep(Math.max(currentStep - 1, 0));
            };

            const handleAddTest = (test) => {
                const existingIndex = tests.findIndex(t => t.id === test.id);
                if (existingIndex >= 0) {
                    const newTests = [...tests];
                    newTests[existingIndex] = test;
                    setTests(newTests);
                } else {
                    setTests([...tests, test]);
                }
            };

            const handleRemoveTest = (testId) => {
                setTests(tests.filter(t => t.id !== testId));
            };

            const handleSave = async () => {
                setSaving(true);
                setSaveError(null);

                try {
                    const payload = {
                        name: suiteData.name,
                        version: suiteData.version,
                        description: suiteData.description || null,
                        defaults: suiteData.defaults,
                        agents: [],
                        tests: tests
                    };

                    const result = await api.post('/suite-definitions', payload);
                    if (onSuccess) {
                        onSuccess(result);
                    }
                } catch (err) {
                    setSaveError(err.message || 'Failed to save suite definition');
                } finally {
                    setSaving(false);
                }
            };

            if (loading) {
                return <SkeletonTestCreator />;
            }

            return (
                <div className="bg-white rounded-lg shadow-lg">
                    {/* Header */}
                    <div className="p-6 border-b flex items-center justify-between">
                        <div>
                            <h2 className="text-xl font-bold text-gray-800">Create Test Suite</h2>
                            <p className="text-sm text-gray-500 mt-1">
                                Define a new test suite to evaluate your agents
                            </p>
                        </div>
                        {onClose && (
                            <button
                                onClick={onClose}
                                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                            >
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        )}
                    </div>

                    {/* Step Indicator */}
                    <div className="px-6 pt-6">
                        <StepIndicator currentStep={currentStep} steps={steps} />
                    </div>

                    {/* Step Content */}
                    <div className="p-6">
                        {currentStep === 0 && (
                            <SuiteDetailsStep
                                suiteData={suiteData}
                                onChange={setSuiteData}
                                errors={errors}
                            />
                        )}

                        {currentStep === 1 && (
                            <TemplateSelectionStep
                                templates={templates}
                                tests={tests}
                                onAddTest={handleAddTest}
                                onRemoveTest={handleRemoveTest}
                                onEditTest={handleAddTest}
                                selectedCategory={selectedCategory}
                                onCategoryChange={setSelectedCategory}
                                loading={false}
                            />
                        )}

                        {currentStep === 2 && (
                            <YAMLPreviewStep
                                suiteData={suiteData}
                                tests={tests}
                                onSave={handleSave}
                                saving={saving}
                                error={saveError}
                            />
                        )}
                    </div>

                    {/* Navigation */}
                    <div className="px-6 pb-6 flex justify-between">
                        <button
                            onClick={handleBack}
                            disabled={currentStep === 0}
                            className="flex items-center gap-2 px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                            </svg>
                            Back
                        </button>

                        {currentStep < steps.length - 1 && (
                            <button
                                onClick={handleNext}
                                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                            >
                                Next
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                </svg>
                            </button>
                        )}
                    </div>
                </div>
            );
        }

        // TestCreatorView - wrapper component for the route
        function TestCreatorView({ onBack, onSuccess }) {
            const handleSuccess = (result) => {
                if (onSuccess) {
                    onSuccess(result);
                }
                if (onBack) {
                    onBack();
                }
            };

            return (
                <div className="max-w-4xl mx-auto">
                    <TestCreatorForm onClose={onBack} onSuccess={handleSuccess} />
                </div>
            );
        }

        // ==================== End Test Creator Form Components ====================

        // Main App component
        function App() {
            const [view, setView] = useState('dashboard');
            const [selectedExecution, setSelectedExecution] = useState(null);
            const [summary, setSummary] = useState(null);
            const [agents, setAgents] = useState([]);
            const [suiteNames, setSuiteNames] = useState([]);
            const [selectedSuite, setSelectedSuite] = useState('');
            const [isLoggedIn, setIsLoggedIn] = useState(!!localStorage.getItem('token'));
            const [loading, setLoading] = useState(true);

            useEffect(() => {
                Promise.all([
                    api.get('/dashboard/summary'),
                    api.get('/agents'),
                    api.get('/suites/names/list')
                ])
                    .then(([sum, ag, names]) => {
                        setSummary(sum);
                        setAgents(ag);
                        setSuiteNames(names);
                        if (names.length > 0) setSelectedSuite(names[0]);
                    })
                    .finally(() => setLoading(false));
            }, []);

            const handleLogout = () => {
                localStorage.removeItem('token');
                setIsLoggedIn(false);
            };

            if (loading) {
                return (
                    <div className="min-h-screen">
                        {/* Header skeleton */}
                        <header className="bg-blue-600 text-white p-4">
                            <div className="container mx-auto flex justify-between items-center">
                                <SkeletonBox className="h-7 w-36 bg-blue-400" />
                                <div className="flex items-center gap-4">
                                    {[1, 2, 3, 4, 5].map((i) => (
                                        <SkeletonBox key={i} className="h-5 w-20 bg-blue-400" />
                                    ))}
                                </div>
                            </div>
                        </header>
                        <main className="container mx-auto p-4">
                            <SkeletonDashboardSummary />
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                <div>
                                    <SkeletonBox className="h-6 w-40 mb-2" />
                                    <SkeletonSuiteList rows={5} />
                                </div>
                                <SkeletonChart />
                            </div>
                        </main>
                    </div>
                );
            }

            return (
                <div className="min-h-screen">
                    {/* Header */}
                    <header className="bg-blue-600 text-white p-4">
                        <div className="container mx-auto flex justify-between items-center">
                            <h1 className="text-xl font-bold">ATP Dashboard</h1>
                            <nav className="flex items-center gap-4">
                                <button
                                    onClick={() => { setView('dashboard'); setSelectedExecution(null); }}
                                    className={`hover:underline ${view === 'dashboard' ? 'font-bold' : ''}`}
                                >
                                    Dashboard
                                </button>
                                <button
                                    onClick={() => { setView('suites'); setSelectedExecution(null); }}
                                    className={`hover:underline ${view === 'suites' ? 'font-bold' : ''}`}
                                >
                                    Suites
                                </button>
                                <button
                                    onClick={() => { setView('compare'); setSelectedExecution(null); }}
                                    className={`hover:underline ${view === 'compare' ? 'font-bold' : ''}`}
                                >
                                    Compare
                                </button>
                                <button
                                    onClick={() => { setView('leaderboard'); setSelectedExecution(null); }}
                                    className={`hover:underline ${view === 'leaderboard' ? 'font-bold' : ''}`}
                                >
                                    Leaderboard
                                </button>
                                <button
                                    onClick={() => { setView('timeline'); setSelectedExecution(null); }}
                                    className={`hover:underline ${view === 'timeline' ? 'font-bold' : ''}`}
                                >
                                    Timeline
                                </button>
                                <button
                                    onClick={() => { setView('create'); setSelectedExecution(null); }}
                                    className={`hover:underline ${view === 'create' ? 'font-bold' : ''} bg-green-500 px-3 py-1 rounded text-sm`}
                                >
                                    + Create
                                </button>
                                {isLoggedIn ? (
                                    <button onClick={handleLogout} className="hover:underline">
                                        Logout
                                    </button>
                                ) : (
                                    <button
                                        onClick={() => setView('login')}
                                        className="hover:underline"
                                    >
                                        Login
                                    </button>
                                )}
                            </nav>
                        </div>
                    </header>

                    {/* Main content */}
                    <main className="container mx-auto p-4">
                        {view === 'login' && (
                            <LoginForm onLogin={() => { setIsLoggedIn(true); setView('dashboard'); }} />
                        )}

                        {view === 'dashboard' && summary && (
                            <ErrorBoundary title="Dashboard Error" message="Unable to display the dashboard.">
                                <DashboardSummary summary={summary} />
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                    <div>
                                        <h2 className="text-lg font-bold mb-2">Recent Executions</h2>
                                        <SuiteList
                                            executions={summary.recent_executions}
                                            onSelect={(id) => { setSelectedExecution(id); setView('suites'); }}
                                        />
                                    </div>
                                    <div>
                                        {selectedSuite && <TrendChart suiteName={selectedSuite} />}
                                    </div>
                                </div>
                            </ErrorBoundary>
                        )}

                        {view === 'suites' && !selectedExecution && summary && (
                            <ErrorBoundary title="Suites Error" message="Unable to display suite list.">
                                <h2 className="text-lg font-bold mb-4">Suite Executions</h2>
                                <SuiteList
                                    executions={summary.recent_executions}
                                    onSelect={setSelectedExecution}
                                />
                            </ErrorBoundary>
                        )}

                        {view === 'suites' && selectedExecution && (
                            <ErrorBoundary title="Suite Details Error" message="Unable to display suite details.">
                                <SuiteDetail
                                    executionId={selectedExecution}
                                    onBack={() => setSelectedExecution(null)}
                                />
                            </ErrorBoundary>
                        )}

                        {view === 'compare' && (
                            <ErrorBoundary title="Comparison Error" message="Unable to display agent comparison.">
                                <h2 className="text-lg font-bold mb-4">Agent Comparison</h2>
                                <div className="mb-4">
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Select Suite:
                                    </label>
                                    <select
                                        value={selectedSuite}
                                        onChange={(e) => setSelectedSuite(e.target.value)}
                                        className="border rounded p-2"
                                    >
                                        {suiteNames.map((name) => (
                                            <option key={name} value={name}>{name}</option>
                                        ))}
                                    </select>
                                </div>
                                {selectedSuite && (
                                    <AgentComparison suiteName={selectedSuite} agents={agents} />
                                )}
                            </ErrorBoundary>
                        )}

                        {view === 'leaderboard' && (
                            <ErrorBoundary title="Leaderboard Error" message="Unable to display the leaderboard.">
                                <h2 className="text-lg font-bold mb-4">Leaderboard Matrix</h2>
                                <div className="mb-4">
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Select Suite:
                                    </label>
                                    <select
                                        value={selectedSuite}
                                        onChange={(e) => setSelectedSuite(e.target.value)}
                                        className="border rounded p-2"
                                    >
                                        {suiteNames.map((name) => (
                                            <option key={name} value={name}>{name}</option>
                                        ))}
                                    </select>
                                </div>
                                {selectedSuite && (
                                    <LeaderboardView suiteName={selectedSuite} />
                                )}
                            </ErrorBoundary>
                        )}

                        {view === 'timeline' && (
                            <ErrorBoundary title="Timeline Error" message="Unable to display the timeline.">
                                <h2 className="text-lg font-bold mb-4">Event Timeline</h2>
                                <div className="mb-4">
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Select Suite:
                                    </label>
                                    <select
                                        value={selectedSuite}
                                        onChange={(e) => setSelectedSuite(e.target.value)}
                                        className="border rounded p-2"
                                    >
                                        {suiteNames.map((name) => (
                                            <option key={name} value={name}>{name}</option>
                                        ))}
                                    </select>
                                </div>
                                {selectedSuite && (
                                    <TimelineView suiteName={selectedSuite} />
                                )}
                            </ErrorBoundary>
                        )}

                        {view === 'create' && (
                            <ErrorBoundary title="Create Suite Error" message="Unable to display the test creator.">
                                <TestCreatorView
                                    onBack={() => setView('dashboard')}
                                    onSuccess={(result) => {
                                        // Refresh suite names after successful creation
                                        api.get('/suites/names/list')
                                            .then(names => {
                                                setSuiteNames(names);
                                                if (result && result.name) {
                                                    setSelectedSuite(result.name);
                                                }
                                            })
                                            .catch(console.error);
                                        setView('dashboard');
                                    }}
                                />
                            </ErrorBoundary>
                        )}

                        {!summary && view !== 'login' && view !== 'leaderboard' && view !== 'timeline' && view !== 'create' && (
                            <div className="text-center py-10">
                                <p className="text-gray-600">
                                    No data available. Run some tests to see results here.
                                </p>
                            </div>
                        )}
                    </main>
                </div>
            );
        }

        // Render the app
        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    </script>
</body>
</html>"""


# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def index():  # pragma: no cover
    """Serve the dashboard frontend."""
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return HTMLResponse(content=create_index_html())


# Mount static files if directory exists
if STATIC_DIR.exists():  # pragma: no cover
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
async def health_check() -> dict[str, str]:  # pragma: no cover
    """Health check endpoint."""
    return {"status": "healthy"}


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    return app


def run_server(  # pragma: no cover
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False,
) -> None:
    """Run the dashboard server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        reload: Whether to enable auto-reload.
    """
    import uvicorn

    uvicorn.run(
        "atp.dashboard.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":  # pragma: no cover
    run_server(reload=True)
