// Reader and timeline questions use the same durable inbox as MDT sheet annotations.
function createAgentQuestions({ getCase, getIdentity, getJSON, postJSON, operationId, onSubmitted }) {
  const { t } = window.brainlensI18n;
  const widgets = [];
  let actions = [], request = 0, fetchedCase = null;

  function render() {
    const active = getCase();
    for (const w of widgets) {
      const context = active && w.getContext();
      w.box.hidden = !active;
      w.send.disabled = !context || w.sending;
      const rows = actions.filter(a => a.case === active && a.source === w.source &&
        (w.source !== "waveform_reader" || (context && a.recording === context.recording && a.run_name === context.run_name)));
      w.thread.replaceChildren(...rows.reverse().map(a => {
        const row = document.createElement("article");
        const question = document.createElement("strong");
        question.textContent = a.comment;
        const meta = document.createElement("small");
        const state = a.status === "resolved" ? t("Resolved") : a.status === "claimed" ? t("In review") : t("Pending");
        const windowLabel = Number.isFinite(a.start_seconds) ? `[${a.start_seconds}, ${a.end_seconds}] s` : "";
        meta.textContent = [a.recording, a.channel, windowLabel, a.marker?.label, state].filter(Boolean).join(" · ");
        row.append(question, meta);
        if (a.agent_reply) {
          const reply = document.createElement("p");
          reply.textContent = a.agent_reply;
          row.append(reply);
        }
        return row;
      }));
    }
  }

  async function refresh(force = true) {
    const caseSlug = getCase();
    render();
    if (!caseSlug || (!force && fetchedCase === caseSlug)) return;
    const seq = ++request;
    try {
      // A busy MDT sheet must not push a reader's question beyond the default 100-action page.
      const inbox = await getJSON("/api/report-actions", { case: caseSlug, limit: 0 });
      if (seq !== request || getCase() !== caseSlug) return;
      fetchedCase = caseSlug;
      actions = (inbox.actions || []).filter(a => widgets.some(w => w.source === a.source));
      render();
    } catch (err) {
      if (seq === request && getCase() === caseSlug) {
        for (const w of widgets) w.status.textContent = err.message;
      }
    }
  }

  function clear() {
    ++request;
    actions = [];
    fetchedCase = null;
    for (const w of widgets) {
      w.epoch++;
      w.input.value = w.status.textContent = "";
      w.sending = false;
      w.retry = null;
      w.box.hidden = true;
      w.thread.replaceChildren();
    }
  }

  function mount(box, getContext) {
    box.append(document.getElementById("askAgentTemplate").content.cloneNode(true));
    const form = box.querySelector("form"), input = form.elements.question;
    const w = { box, getContext, source: box.dataset.source, input, send: form.querySelector("button"),
      status: box.querySelector("[role=status]"), thread: box.querySelector(".ask-thread"), epoch: 0 };
    // Templates are inert when the initial language is applied to the document.
    w.send.textContent = t("Ask agent");
    input.placeholder = t("Ask about this view…");
    input.setAttribute("aria-label", t("Question for the agent"));
    widgets.push(w);
    form.addEventListener("submit", async e => {
      e.preventDefault();
      const context = getContext(), comment = input.value.trim();
      if (!context || getCase() !== context.case || !comment || w.sending) return;
      const payload = { ...context, ...getIdentity(), action: "ask_agent", comment };
      const key = JSON.stringify(payload);
      if (w.retry?.key !== key) w.retry = { key, id: operationId() };
      const epoch = w.epoch;
      w.sending = w.send.disabled = true;
      w.status.textContent = "";
      try {
        const { action } = await postJSON("/api/report-action", { ...payload, operation_id: w.retry.id });
        if (epoch !== w.epoch) return;
        if (input.value.trim() === comment) input.value = "";
        w.retry = null;
        if (!actions.some(a => a.id === action.id)) actions.push(action);
        await refresh();
        onSubmitted();
      } catch (err) {
        if (epoch === w.epoch) w.status.textContent = err.message;
      } finally {
        if (epoch === w.epoch) { w.sending = false; render(); }
      }
    });
  }

  return { mount, clear, refresh, render, editing: () => widgets.some(w => w.input.value.trim() && w.box.offsetWidth) };
}
