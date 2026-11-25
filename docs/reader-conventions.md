# Waveform reader display conventions

## Sensitivity and screen calibration

**Sensitivity → Auto** keeps the original amplitude scale. The reader estimates a stable µV/channel baseline for the recording, montage and display filter. Use the relative gain control or the per-channel 2× boost in this mode.

Choose a sensitivity from **1 to 1000 µV/mm** to use the same physical amplitude scale on every visible channel. At 100 µV/mm, a 100 µV deflection spans 1 mm. Row height changes the distance between channel baselines. The +/− buttons, keyboard shortcuts and Alt + wheel step through the selected mode's scale. Returning to Auto restores the relative gain and channel boosts.

Open **Calibrate screen**, measure the bar end to end with a ruler, enter the measured length in millimetres and save. The reader stores CSS pixels per millimetre in localStorage for this browser and workbench address. Calibrate again after moving to a different display or changing browser zoom or display scaling. **Use 96 dpi** restores the default assumption of 96 CSS pixels per inch. The reader labels the active calibration state.

The pulse under the waveform uses the trace's vertical scale and the plotted time axis. Its labels give the amplitude and duration. It spans one second on pages of at least one second and a labelled shorter interval on subsecond pages. In Auto mode the pulse uses the global gain before individual channel boosts.

## Timebase and paging

**Fit** uses the duration selector and defaults to 8 seconds. Choose **15, 30 or 60 mm/s** to derive the window from the width of the waveform plot and the current calibration. The window adjusts when the plot width changes. The readout shows the effective speed, including when the window reaches the 30-second server limit, the 0.25-second minimum, or the length of a shorter recording. Returning to Fit restores the previous duration selection. Ctrl/⌘ + wheel returns to Fit and zooms around the pointer.

**Page step** defaults to 85% of the window, leaving 15% overlap. Choose **Full page** for consecutive windows. The last page stops at the end of the recording, so it can overlap the preceding page.

## Display filters

The existing band presets fill in **LFF** (high-pass cutoff) and **HFF** (low-pass cutoff), in Hz. Edit either field to enter a custom band. Leave one field empty for a single-sided filter, or clear both for broadband. The Notch checkbox uses the recording's line frequency and its second harmonic where supported.

The status line reports the effective filters. A cutoff marked **varies** differs across returned traces. The reader reports when it limits HFF or falls back to broadband. Each trace uses its contributing channels' native sampling rates and acquisition bandwidth, including both contacts in an adjacent bipolar pair.

`GET /api/multichannel` and `compute_multichannel_batch` accept optional `display_lff` and `display_hff`. Supply finite, positive Hz values, with LFF below HFF when both are present. Either explicit edge takes precedence over `display_filter`. Omitting both uses the existing preset. Effective edges are returned in `display_filter_band_hz` on each trace, with a null edge for a single-sided filter. The upper limit is the lower of 92% of effective Nyquist and the acquisition low-pass. A trace with no usable requested band displays broadband.

These controls affect the multichannel display only. Saved analysis, HFO counts, review judgments and exported clinical documents retain their existing meaning.

## Keyboard navigation

**PageUp / PageDown** move one selected page step. **Home / End** show the first or last window. **?** or the toolbar help button opens the full shortcut list, including verdict keys, waveform tools, gain and wheel gestures. **Esc** closes help. Reader navigation leaves text entry and open dialogs in control of the keyboard.
