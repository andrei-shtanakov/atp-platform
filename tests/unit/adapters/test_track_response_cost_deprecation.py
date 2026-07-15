"""track_response_cost is deprecated by the 003e UsageCapture seam."""

import pytest

from atp.adapters.base import track_response_cost
from atp.protocol import ATPResponse, ResponseStatus


@pytest.mark.anyio
async def test_track_response_cost_warns_deprecation() -> None:
    response = ATPResponse(task_id="t", status=ResponseStatus.COMPLETED)
    with pytest.warns(DeprecationWarning, match="UsageCapture"):
        await track_response_cost(response, provider="anthropic")
