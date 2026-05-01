from __future__ import annotations

import json

from spark_runtime.telemetry import TelemetryConfig, TelemetryRecorder


def test_telemetry_recorder_writes_jsonl(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    recorder = TelemetryRecorder(
        TelemetryConfig(
            jsonl_path=path,
            nvtx=False,
            cuda_events=False,
            decode_step_limit=8,
        )
    )

    recorder.emit({"event": "engine_generate", "request_id": "abc", "tokens": 2})

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [
        {
            "timestamp": rows[0]["timestamp"],
            "event": "engine_generate",
            "request_id": "abc",
            "tokens": 2,
        }
    ]
