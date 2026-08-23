"""Send replay heartbeats to a running AI server and print watcher messages."""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import websockets


async def run(url: str, session: int, at_time: str, duration: float) -> None:
    async with websockets.connect(url, max_size=None) as websocket:
        started = time.monotonic()
        next_heartbeat = started
        while time.monotonic() - started < duration:
            now = time.monotonic()
            if now >= next_heartbeat:
                await websocket.send(json.dumps({
                    "type": "replay_state",
                    "session_key": session,
                    "at_time": at_time,
                    "is_playing": True,
                    "playback_speed": 1.0,
                }))
                next_heartbeat = now + 0.5
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            print(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8001/ws")
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--at-time", required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.session, args.at_time, args.duration))


if __name__ == "__main__":
    main()
