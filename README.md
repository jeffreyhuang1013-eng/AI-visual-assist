# AI-visual-assist
The idea is to use a Jetson Orin Nano and an OAK-D Lite depth camera to recognize nearby objects, estimate distance, help users locate specific items, and provide spoken feedback through earphones.

## V1 prototype

`v1_assistant.py` provides a minimal end-to-end V1 flow:

1. Real-time object detection loop (`VisualAssistV1.run`, pluggable detector)
2. Depth-based distance estimation (`DepthEstimator`)
3. Voice command input (`VoiceCommandInput`)
4. Spoken response output (`Speaker`)
5. Local memory for recently seen objects (`RecentObjectMemory`)

Python version: 3.9+.

Run locally:

```bash
python v1_assistant.py
```

Run tests:

```bash
python -m unittest discover -s tests -q
```
