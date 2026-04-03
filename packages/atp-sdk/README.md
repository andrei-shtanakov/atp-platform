# atp-sdk

Python SDK for the [ATP (Agent Test Platform)](https://github.com/andrei-shtanakov/atp-platform) benchmark platform.

## Installation

```bash
pip install atp-platform-sdk
```

## Quick Start

```python
from atp_sdk import ATPClient

client = ATPClient(platform_url="https://atp.pr0sto.space")
client.login()  # GitHub Device Flow — opens browser

benchmarks = client.list_benchmarks()
run = client.start_run(benchmarks[0].id, agent_name="my-agent")

for task in run:
    response = my_agent(task)  # your agent logic
    run.submit(response, task_index=task["metadata"]["task_index"])

print(run.status())
print(run.leaderboard())
```

## Authentication

Three ways to authenticate (checked in order):

1. **Explicit token**: `ATPClient(token="...")`
2. **Environment variable**: `ATP_TOKEN=...`
3. **Saved token**: `client.login()` saves to `~/.atp/config.json`

## License

MIT
