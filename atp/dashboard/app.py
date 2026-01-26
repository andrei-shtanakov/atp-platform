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
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div id="root"></div>
    <script type="text/babel">
        const { useState, useEffect, useCallback } = React;

        // API helper
        const api = {
            async get(path) {
                const token = localStorage.getItem('token');
                const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
                const res = await fetch(`/api${path}`, { headers });
                if (!res.ok) throw new Error(`API error: ${res.status}`);
                return res.json();
            },
            async post(path, data) {
                const token = localStorage.getItem('token');
                const headers = {
                    'Content-Type': 'application/json',
                    ...(token && { 'Authorization': `Bearer ${token}` })
                };
                const res = await fetch(`/api${path}`, {
                    method: 'POST',
                    headers,
                    body: JSON.stringify(data)
                });
                if (!res.ok) throw new Error(`API error: ${res.status}`);
                return res.json();
            }
        };

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

            if (loading) return <p>Loading...</p>;
            if (!execution) return <p>Execution not found</p>;

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

        // Step Comparison component - displays event list for a single agent
        function StepComparison({ agentDetail }) {
            const [eventFilter, setEventFilter] = useState('all');

            if (!agentDetail) {
                return (
                    <div className="bg-gray-50 p-4 rounded text-center text-gray-500">
                        No execution data available
                    </div>
                );
            }

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

                    {/* Event filters */}
                    <div className="p-2 border-b flex flex-wrap gap-1">
                        <button
                            onClick={() => setEventFilter('all')}
                            className={`px-2 py-1 text-xs rounded ${eventFilter === 'all' ? 'bg-gray-800 text-white' : 'bg-gray-100 hover:bg-gray-200'}`}
                        >
                            All ({agentDetail.events?.length || 0})
                        </button>
                        {Object.entries(EVENT_COLORS).map(([type, colors]) => {
                            const count = agentDetail.events?.filter(e => e.event_type === type).length || 0;
                            if (count === 0) return null;
                            return (
                                <button
                                    key={type}
                                    onClick={() => setEventFilter(type)}
                                    className={`px-2 py-1 text-xs rounded ${
                                        eventFilter === type
                                            ? `${colors.bg} ${colors.text} ring-2 ring-offset-1`
                                            : 'bg-gray-100 hover:bg-gray-200'
                                    }`}
                                >
                                    {type.replace('_', ' ')} ({count})
                                </button>
                            );
                        })}
                    </div>

                    {/* Event list */}
                    <div className="p-4 max-h-96 overflow-y-auto">
                        {filteredEvents && filteredEvents.length > 0 ? (
                            filteredEvents.map((event, idx) => (
                                <EventItem key={idx} event={event} sequence={event.sequence} />
                            ))
                        ) : (
                            <p className="text-gray-500 text-center py-4">No events to display</p>
                        )}
                    </div>
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
                    <div className="flex items-center justify-center py-12">
                        <div className="text-center">
                            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent mb-4"></div>
                            <p className="text-gray-600">Loading comparison data...</p>
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
                            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
                                {error}
                            </div>
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

                    {/* Side-by-side comparison columns */}
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

        // TimelineRow component - single agent timeline visualization
        function TimelineRow({ timeline, totalDurationMs, zoomLevel, eventFilter, onEventClick }) {
            const [hoveredEvent, setHoveredEvent] = useState(null);
            const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
            const rowRef = React.useRef(null);

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

            return (
                <div className="mb-4 last:mb-0">
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

                    {/* Timeline track */}
                    <div
                        ref={rowRef}
                        className="relative h-10 bg-gray-100 rounded border border-gray-200 overflow-hidden"
                    >
                        {filteredEvents.map((event, idx) => (
                            <EventMarker
                                key={`${event.sequence}-${idx}`}
                                event={event}
                                totalDurationMs={totalDurationMs}
                                zoomLevel={zoomLevel}
                                onHover={handleHover}
                                onLeave={() => setHoveredEvent(null)}
                                onClick={onEventClick}
                            />
                        ))}
                        {hoveredEvent && (
                            <EventTooltip event={hoveredEvent} position={hoverPosition} />
                        )}
                    </div>
                </div>
            );
        }

        // EventDetailPanel component - full event details panel
        function EventDetailPanel({ event, onClose }) {
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
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] flex flex-col">
                        {/* Header */}
                        <div className={`p-4 border-b ${colors.bg} rounded-t-lg flex items-center justify-between`}>
                            <div className="flex items-center gap-3">
                                <span className={`w-8 h-8 rounded-full ${colors.border.replace('border', 'bg').replace('400', '200')} ${colors.text} flex items-center justify-center text-lg font-bold`}>
                                    {colors.icon}
                                </span>
                                <div>
                                    <h3 className={`font-bold text-lg ${colors.text}`}>
                                        {event.event_type.replace('_', ' ').toUpperCase()}
                                    </h3>
                                    <p className="text-sm text-gray-500">Event #{event.sequence + 1}</p>
                                </div>
                            </div>
                            <button
                                onClick={onClose}
                                className="p-2 hover:bg-gray-200 rounded-full transition-colors"
                            >
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>

                        {/* Content */}
                        <div className="p-4 overflow-y-auto flex-1">
                            {/* Summary */}
                            <div className="mb-4">
                                <h4 className="text-sm font-semibold text-gray-500 mb-1">Summary</h4>
                                <p className="text-gray-700">{event.summary}</p>
                            </div>

                            {/* Timing */}
                            <div className="mb-4 grid grid-cols-2 gap-4">
                                <div>
                                    <h4 className="text-sm font-semibold text-gray-500 mb-1">Start Time</h4>
                                    <p className="text-gray-700">{formatTime(event.relative_time_ms)}</p>
                                </div>
                                {event.duration_ms && (
                                    <div>
                                        <h4 className="text-sm font-semibold text-gray-500 mb-1">Duration</h4>
                                        <p className="text-gray-700">{formatTime(event.duration_ms)}</p>
                                    </div>
                                )}
                            </div>

                            {/* Raw data */}
                            <div>
                                <h4 className="text-sm font-semibold text-gray-500 mb-1">Event Data</h4>
                                <div className="bg-gray-50 rounded p-3 overflow-x-auto">
                                    <pre className="text-xs font-mono whitespace-pre-wrap text-gray-700">
                                        {JSON.stringify(event.data, null, 2)}
                                    </pre>
                                </div>
                            </div>
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

            // Event type counts
            const eventCounts = timelineData
                ? timelineData.timelines.reduce((acc, timeline) => {
                    timeline.events.forEach(e => {
                        acc[e.event_type] = (acc[e.event_type] || 0) + 1;
                    });
                    return acc;
                }, {})
                : {};

            // Loading state
            if (loading) {
                return (
                    <div className="flex items-center justify-center py-12">
                        <div className="text-center">
                            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent mb-4"></div>
                            <p className="text-gray-600">Loading timeline data...</p>
                        </div>
                    </div>
                );
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
                            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
                                {error}
                            </div>
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
                            <div className="flex flex-wrap gap-1">
                                <button
                                    onClick={() => setEventFilter('all')}
                                    className={`px-2 py-1 text-xs rounded ${eventFilter === 'all' ? 'bg-gray-800 text-white' : 'bg-gray-100 hover:bg-gray-200'}`}
                                >
                                    All
                                </button>
                                {Object.entries(EVENT_COLORS).map(([type, colors]) => {
                                    const count = eventCounts[type] || 0;
                                    if (count === 0) return null;
                                    return (
                                        <button
                                            key={type}
                                            onClick={() => setEventFilter(type)}
                                            className={`px-2 py-1 text-xs rounded ${
                                                eventFilter === type
                                                    ? `${colors.bg} ${colors.text} ring-2 ring-offset-1`
                                                    : 'bg-gray-100 hover:bg-gray-200'
                                            }`}
                                        >
                                            {type.replace('_', ' ')} ({count})
                                        </button>
                                    );
                                })}
                            </div>
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
                        <div className="mt-4 overflow-x-auto">
                            <div style={{ width: `${100 * zoomLevel}%`, minWidth: '100%' }}>
                                {timelineData.timelines.map((timeline, idx) => (
                                    <TimelineRow
                                        key={timeline.agent_name}
                                        timeline={timeline}
                                        totalDurationMs={maxDuration}
                                        zoomLevel={zoomLevel}
                                        eventFilter={eventFilter}
                                        onEventClick={setSelectedEvent}
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
                    <div className="flex items-center justify-center py-12">
                        <div className="text-center">
                            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent mb-4"></div>
                            <p className="text-gray-600">Loading...</p>
                        </div>
                    </div>
                );
            }

            return (
                <TimelineContainer
                    suiteName={suiteName}
                    testId=""
                    agents={[]}
                    availableAgents={agents}
                    onBack={onBack}
                    tests={tests}
                />
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
                return (
                    <div className="flex items-center justify-center py-12">
                        <div className="text-center">
                            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent mb-4"></div>
                            <p className="text-gray-600">Loading leaderboard matrix...</p>
                        </div>
                    </div>
                );
            }

            if (error) {
                return (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
                        <p className="font-semibold">Error loading leaderboard</p>
                        <p className="text-sm">{error}</p>
                        <button
                            onClick={loadMatrix}
                            className="mt-2 text-sm text-red-600 hover:text-red-800 underline"
                        >
                            Try again
                        </button>
                    </div>
                );
            }

            return (
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
                    <div className="flex items-center justify-center h-screen">
                        <p className="text-xl">Loading...</p>
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
                            <>
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
                            </>
                        )}

                        {view === 'suites' && !selectedExecution && summary && (
                            <>
                                <h2 className="text-lg font-bold mb-4">Suite Executions</h2>
                                <SuiteList
                                    executions={summary.recent_executions}
                                    onSelect={setSelectedExecution}
                                />
                            </>
                        )}

                        {view === 'suites' && selectedExecution && (
                            <SuiteDetail
                                executionId={selectedExecution}
                                onBack={() => setSelectedExecution(null)}
                            />
                        )}

                        {view === 'compare' && (
                            <>
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
                            </>
                        )}

                        {view === 'leaderboard' && (
                            <>
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
                            </>
                        )}

                        {view === 'timeline' && (
                            <>
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
                            </>
                        )}

                        {!summary && view !== 'login' && view !== 'leaderboard' && view !== 'timeline' && (
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
