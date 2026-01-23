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

        // Agent comparison component
        function AgentComparison({ suiteName, agents }) {
            const [comparison, setComparison] = useState(null);
            const [selectedAgents, setSelectedAgents] = useState([]);
            const [loading, setLoading] = useState(false);

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
                    <h3 className="font-bold mb-4">Agent Comparison: {comparison.suite_name}</h3>
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

                        {!summary && view !== 'login' && (
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
