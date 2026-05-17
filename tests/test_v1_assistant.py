import unittest

from v1_assistant import (
    ConsoleSpeaker,
    DetectedObject,
    QueueVoiceCommandInput,
    RecentObjectMemory,
    SimulatedDepthEstimator,
    SimulatedObjectDetector,
    VisualAssistV1,
)


class VisualAssistV1Tests(unittest.TestCase):
    def test_what_do_you_see_reports_detected_objects_with_distance(self) -> None:
        detector = SimulatedObjectDetector(
            frames=[[DetectedObject("chair", 0.9), DetectedObject("bottle", 0.8)]]
        )
        command_input = QueueVoiceCommandInput(["what do you see"])
        speaker = ConsoleSpeaker()
        app = VisualAssistV1(
            detector=detector,
            depth_estimator=SimulatedDepthEstimator(),
            command_input=command_input,
            speaker=speaker,
            memory=RecentObjectMemory(),
        )

        app.process_frame()

        self.assertEqual(1, len(speaker.utterances))
        self.assertIn("chair at 1.6 meters", speaker.utterances[0])
        self.assertIn("bottle at 0.8 meters", speaker.utterances[0])

    def test_where_is_uses_recent_memory(self) -> None:
        detector = SimulatedObjectDetector(
            frames=[[DetectedObject("phone", 0.9)], [DetectedObject("laptop", 0.95)]]
        )
        command_input = QueueVoiceCommandInput(["", "where is phone"])
        speaker = ConsoleSpeaker()
        app = VisualAssistV1(
            detector=detector,
            depth_estimator=SimulatedDepthEstimator(),
            command_input=command_input,
            speaker=speaker,
            memory=RecentObjectMemory(),
        )

        app.process_frame()
        app.process_frame()

        self.assertEqual("phone is about 0.7 meters away.", speaker.utterances[0])

    def test_recent_objects_returns_most_recent_unique_names(self) -> None:
        detector = SimulatedObjectDetector(
            frames=[
                [DetectedObject("chair", 0.9)],
                [DetectedObject("chair", 0.9), DetectedObject("table", 0.9)],
            ]
        )
        command_input = QueueVoiceCommandInput(["", "recent objects"])
        speaker = ConsoleSpeaker()
        app = VisualAssistV1(
            detector=detector,
            depth_estimator=SimulatedDepthEstimator(),
            command_input=command_input,
            speaker=speaker,
            memory=RecentObjectMemory(),
        )

        app.process_frame()
        app.process_frame()

        self.assertEqual("Recently seen objects: table, chair.", speaker.utterances[0])


if __name__ == "__main__":
    unittest.main()
