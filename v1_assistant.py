from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Iterable, List, Optional, Protocol, Sequence, Tuple
import time

WHERE_IS_PREFIX = "where is "
DEMO_DURATION_SECONDS = 1.6


@dataclass(frozen=True)
class DetectedObject:
    name: str
    confidence: float


@dataclass(frozen=True)
class ObservedObject:
    name: str
    confidence: float
    distance_m: Optional[float]
    seen_at: datetime


class ObjectDetector(Protocol):
    def detect(self) -> Sequence[DetectedObject]:
        ...


class DepthEstimator(Protocol):
    def estimate_meters(self, obj: DetectedObject) -> Optional[float]:
        ...


class VoiceCommandInput(Protocol):
    def listen_non_blocking(self) -> Optional[str]:
        ...


class Speaker(Protocol):
    def speak(self, text: str) -> None:
        ...


class RecentObjectMemory:
    def __init__(self, capacity: int = 20) -> None:
        self._memory: Deque[ObservedObject] = deque(maxlen=capacity)

    def remember(self, objects: Sequence[ObservedObject]) -> None:
        self._memory.extend(objects)

    def most_recent(self, name: str) -> Optional[ObservedObject]:
        normalized = name.strip().lower()
        for obj in reversed(self._memory):
            if obj.name.lower() == normalized:
                return obj
        return None

    def recent_names(self, limit: int = 5) -> List[str]:
        names: List[str] = []
        seen = set()
        for obj in reversed(self._memory):
            lowered = obj.name.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            names.append(obj.name)
            if len(names) >= limit:
                break
        return names


class QueueVoiceCommandInput:
    def __init__(self, commands: Optional[Iterable[str]] = None) -> None:
        self._commands: Deque[str] = deque(commands or [])

    def listen_non_blocking(self) -> Optional[str]:
        if not self._commands:
            return None
        return self._commands.popleft()


class ConsoleSpeaker:
    def __init__(self) -> None:
        self.utterances: List[str] = []

    def speak(self, text: str) -> None:
        self.utterances.append(text)
        print(text)


class SimulatedObjectDetector:
    def __init__(self, frames: Optional[Sequence[Sequence[DetectedObject]]] = None) -> None:
        self._frames = list(
            frames
            or [
                [DetectedObject("chair", 0.94), DetectedObject("table", 0.91)],
                [DetectedObject("bottle", 0.90)],
                [DetectedObject("laptop", 0.92), DetectedObject("phone", 0.88)],
            ]
        )
        self._index = 0

    def detect(self) -> Sequence[DetectedObject]:
        frame = self._frames[self._index]
        self._index = (self._index + 1) % len(self._frames)
        return frame


class SimulatedDepthEstimator:
    def __init__(self, distances_m: Optional[dict[str, float]] = None) -> None:
        if distances_m is None:
            self._distances_m = {
                "chair": 1.6,
                "table": 2.2,
                "bottle": 0.8,
                "laptop": 1.1,
                "phone": 0.7,
            }
        else:
            self._distances_m = distances_m

    def estimate_meters(self, obj: DetectedObject) -> Optional[float]:
        return self._distances_m.get(obj.name.lower())


class VisualAssistV1:
    def __init__(
        self,
        detector: ObjectDetector,
        depth_estimator: DepthEstimator,
        command_input: VoiceCommandInput,
        speaker: Speaker,
        memory: Optional[RecentObjectMemory] = None,
    ) -> None:
        self.detector = detector
        self.depth_estimator = depth_estimator
        self.command_input = command_input
        self.speaker = speaker
        self.memory = memory or RecentObjectMemory()

    def process_frame(self) -> Tuple[Sequence[ObservedObject], Optional[str]]:
        now = datetime.now(tz=timezone.utc)
        detected = self.detector.detect()
        observed = [
            ObservedObject(
                name=obj.name,
                confidence=obj.confidence,
                distance_m=self.depth_estimator.estimate_meters(obj),
                seen_at=now,
            )
            for obj in detected
        ]
        self.memory.remember(observed)
        command = self.command_input.listen_non_blocking()
        if command:
            self.speaker.speak(self._respond_to_command(command, observed))
        return observed, command

    def run(self, seconds: float = 5.0, hz: float = 2.0) -> None:
        """Run the real-time detection loop for a fixed duration."""
        if seconds <= 0:
            raise ValueError(f"seconds must be greater than 0, got {seconds}")
        if hz <= 0:
            raise ValueError(f"hz must be greater than 0, got {hz}")
        interval = 1.0 / hz
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            loop_start = time.monotonic()
            self.process_frame()
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, interval - elapsed))

    def _respond_to_command(self, command: str, observed: Sequence[ObservedObject]) -> str:
        normalized = command.strip().lower()
        if normalized == "what do you see":
            if not observed:
                return "I do not detect any objects right now."
            details = ", ".join(_describe_object(obj) for obj in observed)
            return f"I see {details}."

        if normalized == "recent objects":
            names = self.memory.recent_names()
            if not names:
                return "I have not seen any objects yet."
            return "Recently seen objects: " + ", ".join(names) + "."

        if normalized.startswith(WHERE_IS_PREFIX):
            target = normalized.removeprefix(WHERE_IS_PREFIX).strip()
            if not target:
                return "Please tell me which object to find."
            found = self.memory.most_recent(target)
            if not found:
                return f"I have not seen {target} recently."
            if found.distance_m is None:
                return f"I saw {found.name}, but I cannot estimate the distance."
            return f"{found.name} is about {found.distance_m:.1f} meters away."

        return (
            "I can respond to 'what do you see', 'where is <object>', or 'recent objects'."
        )


def _describe_object(obj: ObservedObject) -> str:
    if obj.distance_m is None:
        return f"{obj.name} (distance unknown)"
    return f"{obj.name} at {obj.distance_m:.1f} meters"


def build_default_app(commands: Optional[Iterable[str]] = None) -> VisualAssistV1:
    return VisualAssistV1(
        detector=SimulatedObjectDetector(),
        depth_estimator=SimulatedDepthEstimator(),
        command_input=QueueVoiceCommandInput(commands),
        speaker=ConsoleSpeaker(),
        memory=RecentObjectMemory(capacity=30),
    )


if __name__ == "__main__":
    app = build_default_app(commands=["what do you see", "where is bottle", "recent objects"])
    app.run(seconds=DEMO_DURATION_SECONDS, hz=2.0)
