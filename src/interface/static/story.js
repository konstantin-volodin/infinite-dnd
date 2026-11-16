    const POLL_MS = 3000;
    const state = { runs: [], scenario: null, runId: null, chapters: [], tail: null, at: null, tailCount: -1 };
    const dom = {
      run: document.getElementById("run"),
      title: document.getElementById("meta-title"),
      box: document.getElementById("chapters-box"),
      count: document.getElementById("chapter-count"),
      list: document.getElementById("chapter-list"),
      reader: document.getElementById("reader"),
    };

    // Position lives in the URL hash: #<scenario>/<run>/<chapter number | tail>
    function readHash() {
      const [scenario, runId, at] = location.hash.slice(1).split("/").map(decodeURIComponent);
      return { scenario, runId, at };
    }

    function go(at) {
      state.at = String(at);
      const hash = `#${[state.scenario, state.runId, state.at].map(encodeURIComponent).join("/")}`;
      history.replaceState(null, "", hash);
      try { localStorage.setItem("story-position", hash); } catch (_) { /* optional */ }
      renderList();
      return render();
    }

    function inline(text) {
      return escapeHtml(text)
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>");
    }

    function paragraphs(prose) {
      return String(prose ?? "").split(/\n\s*\n/).filter((p) => p.trim())
        .map((p) => `<p>${inline(p.trim())}</p>`).join("");
    }

    function eventRows(events) {
      return events.map((e) => `
        <li><span class="ev-meta">${escapeHtml(e.location)}${(e.characters || []).length ? ` · ${escapeHtml(e.characters.join(", "))}` : ""}</span>
        <span class="ev-text">${escapeHtml(e.text)}</span></li>`).join("");
    }

    const api = (suffix = "") => `/api/story/${encodeURIComponent(state.scenario)}/${encodeURIComponent(state.runId)}${suffix}`;

    function pager(n) {
      const prev = n > 1 ? `<button type="button" data-go="${n - 1}">‹ Previous</button>` : "<span></span>";
      const next = n < state.chapters.length ? `<button type="button" data-go="${n + 1}">Next ›</button>`
        : `<button type="button" data-go="tail">Unfolding now ›</button>`;
      return `<nav class="pager" aria-label="Chapter navigation">${prev}${next}</nav>`;
    }

    async function renderChapter(n) {
      const chapter = await fetchJson(api(`/${n}`));
      if (state.at !== String(n)) return;
      const events = chapter.events || [];
      dom.reader.innerHTML = `
        <article class="chapter">
          <p class="kicker">Chapter ${n}</p>
          <h1>${escapeHtml(chapter.title)}</h1>
          ${paragraphs(chapter.prose)}
          <details class="raw"><summary>Raw events <span class="count">${events.length}</span></summary>
            <ol>${eventRows(events)}</ol></details>
        </article>${pager(n)}`;
      window.scrollTo(0, 0);
    }

    function renderTail() {
      const events = state.tail?.events || [];
      state.tailCount = events.length;
      const last = state.chapters.length;
      const prev = last
        ? `<nav class="pager" aria-label="Chapter navigation"><button type="button" data-go="${last}">‹ Previous</button><span></span></nav>` : "";
      dom.reader.innerHTML = `
        <article class="chapter">
          <p class="kicker">${last ? "After the last chapter" : "Story so far"}</p>
          <h1>Unfolding now</h1>
          ${events.length ? `<ol class="tail">${eventRows(events)}</ol>` : '<p class="empty">Nothing has happened yet.</p>'}
        </article>${prev}`;
    }

    async function render() {
      if (!state.scenario) {
        dom.reader.innerHTML = '<p class="empty">Run a game to produce a story.</p>';
      } else if (state.at === "tail") {
        renderTail();
        window.scrollTo(0, 0);
      } else {
        await renderChapter(Number(state.at));
      }
    }

    function renderList() {
      const item = (at, label, title) => `
        <li><button type="button" data-go="${at}"${String(at) === state.at ? ' aria-current="true"' : ""}>
          <span class="num">${label}</span>${escapeHtml(title)}</button></li>`;
      dom.count.textContent = state.chapters.length;
      dom.list.innerHTML = state.chapters.map((c) => item(c.number, c.number, c.title)).join("")
        + item("tail", "…", "Unfolding now");
    }

    function renderRuns() {
      dom.run.innerHTML = state.runs.map((r) => `
        <option value="${escapeHtml(runKey(r.scenario, r.run_id))}" ${r.scenario === state.scenario && r.run_id === state.runId ? "selected" : ""}>
          ${escapeHtml(r.title)} · ${escapeHtml(r.run_id)}
        </option>`).join("");
    }

    async function loadRun(at) {
      const run = state.runs.find((r) => r.scenario === state.scenario && r.run_id === state.runId);
      dom.title.textContent = run ? run.title : "";
      state.chapters = (await fetchJson(api())).chapters;
      state.tail = await fetchJson(api("/tail"));
      const valid = at === "tail" || state.chapters.some((c) => String(c.number) === at);
      await go(valid ? at : state.chapters.length ? 1 : "tail");
    }

    async function init() {
      state.runs = await fetchJson("/api/runs");
      if (!location.hash) {
        try { location.hash = localStorage.getItem("story-position") || ""; } catch (_) { /* optional */ }
      }
      let { scenario, runId, at } = readHash();
      if (!state.runs.some((r) => r.scenario === scenario && r.run_id === runId)) {
        scenario = state.runs[0]?.scenario;
        runId = state.runs[0]?.run_id;
        at = undefined;
      }
      state.scenario = scenario;
      state.runId = runId;
      renderRuns();
      if (scenario) await loadRun(at);
      else await render();
    }

    // Poll only the cheap tail endpoint; reload the chapter list when a chapter appears.
    // Re-render only what changed, so scroll position survives.
    async function poll() {
      if (document.hidden || !state.scenario) return;
      try {
        const tail = await fetchJson(api("/tail"));
        const grew = tail.chapter_count !== state.chapters.length;
        state.tail = tail;
        if (grew) {
          state.chapters = (await fetchJson(api())).chapters;
          renderList();
        }
        if (state.at === "tail" && (grew || tail.events.length !== state.tailCount)) renderTail();
      } catch (_) { /* snapshot mid-write; retry next poll */ }
    }

    document.addEventListener("click", (event) => {
      const target = event.target.closest("[data-go]");
      if (target) go(target.dataset.go).catch(() => {});
    });
    dom.run.addEventListener("change", () => {
      const sep = dom.run.value.indexOf("/");
      state.scenario = dom.run.value.slice(0, sep);
      state.runId = dom.run.value.slice(sep + 1);
      loadRun(undefined).catch(() => {});
    });
    if (matchMedia("(max-width: 800px)").matches) dom.box.open = false;
    init().catch(() => { dom.reader.innerHTML = '<p class="empty">Could not load the story.</p>'; });
    setInterval(poll, POLL_MS);
