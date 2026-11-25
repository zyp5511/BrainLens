# BrainLens 1.0 Launch Video — Voice-Over Script

- **Target runtime:** ~84 s (leave 1:20–1:24 silent on the end card)
- **Lines:** the spoken lines are the on-screen subtitles, word for word, so the silent cut and the voiced cut tell the same story.
- **Delivery:** calm, dry, unhurried; plain words a non-specialist can follow. No sales cadence.
- **Audio muxing:** save the recording as `build/voiceover.wav` (48 kHz WAV recommended) and re-run `python docs/demo/record_launch.py --stage render`. When `build/voiceover.wav` is present, the render stage muxes AAC audio (192 kbps) into `build/brainlens_1_0_launch.mp4` and pads silence through the end card.

---

## Script with Timecodes

| Timecode | On screen | Line (spoken and subtitled) |
| :--- | :--- | :--- |
| **0:00–0:04** | HUP134 case sheet, LB10 highlighted | Before epilepsy surgery, doctors review signals from electrodes inside the brain. |
| **0:04–0:09** | Punch-in: the question being typed | The doctor asks the AI: does this LB10 signal come back in the next recording? |
| **0:09–0:12** | Reader jumps by itself to run-02 / LB10 | The AI checks the next recording and opens the reader right at the match. |
| **0:12–0:16** | Punch-in: the ~230 Hz burst | There it is: a short burst of fast activity, around 230 Hz. |
| **0:16–0:22** | Punch-in: the agent's reply in the inbox | Its answer: yes, twice. But its own artifact model doubts them, so it asks the doctor to look. |
| **0:22–0:25** | Title card | BrainLens. AI co-review for EEG. |
| **0:25–0:30** | Case 1 · Scalp EEG · CHB-MIT: admission timeline | About 40 hours of monitoring, split across 42 files, laid out on one timeline. |
| **0:30–0:33** | Reader at the start of the first file | Instead of scrolling through 40 hours, the doctor just asks. |
| **0:33–0:36** | Punch-in: "Where are the seizures?" typed | *(no line; let the typing read)* |
| **0:36–0:41** | Reader jumps to the first onset | The AI finds the seven seizures marked by experts and opens the first one. |
| **0:41–0:44** | Punch-in: the agent's reply beside the trace | *(no line; let the reply read)* |
| **0:44–0:46** | Onset on the double-banana montage | The first seizure onset, on the standard scalp montage. |
| **0:46–0:51** | Trend view | Trends show the whole recording at a glance, for quick navigation. |
| **0:51–0:55** | Case 2 · SEEG · OpenNeuro HUP134: overview | 1,150 automatic detections. 1,147 of them hit many electrodes at the same moment. |
| **0:55–1:00** | A burst across many channels | Bursts like this one are set aside, not ranked as a focus. |
| **1:00–1:05** | LB10 beside its quiet control | That leaves three local candidates. LB10 sits beside a quiet stretch, on the same scale. |
| **1:05–1:08** | Punch-in: verdict panel, key 3 | The AI calls it artifact. The doctor isn't sure and marks it uncertain. |
| **1:08–1:11** | Both entries shown | Both calls are kept, each labeled with who made it. |
| **1:11–1:14** | Click on MDT slides | One click turns the review into slides for the surgery conference. |
| **1:14–1:20** | The exported slides | The same evidence and both verdicts, ready to present. |
| **1:20–1:24** | End card | *(silent)* |
