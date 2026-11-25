// Reader geometry is in CSS pixels. The canvas transform handles backing-store pixels.
function createReaderScales() {
  const { t } = window.brainlensI18n;
  const byId = (id) => document.getElementById(id);
  const sensitivity = byId("readerSensitivitySelect");
  const levels = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 1000];
  for (const level of levels) sensitivity.add(new Option(level + " µV/mm", String(level)));

  const storageKey = "brainlens_reader_px_per_mm";
  let calibratedPxPerMm = null;
  try {
    const saved = Number(localStorage.getItem(storageKey));
    if (Number.isFinite(saved) && saved > 0) calibratedPxPerMm = saved;
  } catch (_) {}
  const pxPerMm = () => calibratedPxPerMm ?? (96 / 25.4);
  const calibrationLabel = () => calibratedPxPerMm ? t("calibrated") : t("screen not calibrated");
  const fixed = () => sensitivity.value !== "auto";

  function bindCalibration(onChange) {
    const dialog = byId("screenCalibrationDialog");
    const length = byId("calibrationLengthMm");
    const ruler = byId("calibrationRuler");
    const save = (ratio) => {
      calibratedPxPerMm = ratio;
      try {
        if (ratio === null) localStorage.removeItem(storageKey);
        else localStorage.setItem(storageKey, String(ratio));
      } catch (_) {}
      dialog.close();
      onChange();
    };
    byId("calibrateScreenBtn").addEventListener("click", () => {
      dialog.showModal();
      length.setCustomValidity("");
      length.value = calibratedPxPerMm ? (ruler.getBoundingClientRect().width / pxPerMm()).toFixed(2) : "";
      length.focus();
      length.select();
    });
    length.addEventListener("input", () => length.setCustomValidity(""));
    byId("calibrationForm").addEventListener("submit", (event) => {
      event.preventDefault();
      const ratio = ruler.getBoundingClientRect().width / length.valueAsNumber;
      if (!Number.isFinite(ratio) || ratio <= 0) {
        length.setCustomValidity(t("Enter a positive length."));
        length.reportValidity();
        return;
      }
      save(ratio);
    });
    byId("resetCalibrationBtn").addEventListener("click", () => save(null));
    byId("cancelCalibrationBtn").addEventListener("click", () => dialog.close());
  }

  function stepSensitivity(gainDirection) {
    const index = levels.indexOf(Number(sensitivity.value));
    sensitivity.value = String(levels[Math.max(0, Math.min(levels.length - 1, index - gainDirection))]);
  }

  function windowAtSpeed(width, recordingDuration) {
    const speed = Number(byId("readerTimebaseSelect").value);
    if (!speed || width <= 0) return null;
    return Math.min(30, recordingDuration, Math.max(0.25, width / pxPerMm() / speed));
  }

  function scaleLabel(uvLane) {
    return fixed() ? sensitivity.value + " µV/mm (" + calibrationLabel() + ")" : Math.round(uvLane) + " µV/ch";
  }

  function drawPulse(ctx, x, y, width, span, pxPerUv) {
    // Pick a readable 1, 2 or 5 × 10^n amplitude. Use a shorter pulse on subsecond pages.
    const target = 16 / pxPerUv;
    const magnitude = 10 ** Math.floor(Math.log10(target));
    const uv = Number((([5, 2, 1].find(n => n * magnitude <= target) || 1) * magnitude).toPrecision(3));
    const seconds = span >= 1 ? 1 : Math.min(span, span >= 0.5 ? 0.5 : 0.25);
    const right = x + width * seconds / span;
    const top = y - uv * pxPerUv;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, top);
    ctx.lineTo(right, top);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.textAlign = "left";
    ctx.fillText(uv + " µV", 8, y - 4);
    ctx.fillText(seconds + " s", 8, y + 7);
  }

  return {
    fixed, stepSensitivity, bindCalibration, windowAtSpeed, scaleLabel, drawPulse,
    uvPerRow: (rowH) => rowH * Number(sensitivity.value) / pxPerMm(),
    timebaseLabel: (width, span) => (width / pxPerMm() / span).toFixed(1) + " mm/s (" + calibrationLabel() + ")",
  };
}
