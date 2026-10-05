    const EVENT_COLOR = {
      game_session_started: "violet",
      game_session_finished: "violet",
      campaign_completed: "green",
      campaign_failed: "ember",
      pc_death: "ember",
      resolved: "green",
      world_snapshot: "amber",
      player_action: "amber",
      agent_run_started: "",
      agent_run_finished: "",
      llm_message: "",
      parse_error: "ember",
      run_error: "ember",
      world_update_rejected: "amber",
    };

    const STORY_EVENTS = new Set([
      "game_session_started", "game_session_finished", "campaign_completed", "campaign_failed", "pc_death",
      "resolved", "world_snapshot", "player_action", "parse_error", "run_error", "world_update_rejected",
    ]);

    const state = {
      files: [],
      session: null,
      search: "",
      detail: "story",
      eventFilter: null,
      labelFilter: null,
      selectedLine: null,
    };

    const dom = {
      fileList: document.getElementById("file-list"),
      eventList: document.getElementById("event-list"),
      labelList: document.getElementById("label-list"),
      logPanel: document.getElementById("log-panel"),
      inspector: document.getElementById("inspector"),
      search: document.getElementById("search"),
      detailSeg: document.getElementById("detail-seg"),
      sessionTitle: document.getElementById("session-title"),
      sessionSub: document.getElementById("session-sub"),
    };

    function formatDateTime(value) {
      if (!value) return "unknown";
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
    }

    function formatDuration(value) {
      if (value === null || value === undefined) return "--";
      return `${Number(value).toFixed(1)}s`;
    }

    async function loadFiles() {
      state.files = await fetchJson("/api/files");
      renderFiles();
      if (!state.files.length) {
        dom.logPanel.innerHTML = '<div class="empty">No log files were found in the selected directory.</div>';
        return;
      }
      await loadSession(state.files[0].name);
    }

    async function loadSession(name) {
      state.session = await fetchJson(`/api/logs/${encodeURIComponent(name)}`);
      state.eventFilter = null;
      state.labelFilter = null;
      state.selectedLine = null;
      renderAll();
    }

    function renderAll() {
      renderFiles();
      renderTopbar();
      renderRailFilters();
      renderRows();
      renderInspector();
    }

    function renderFiles() {
      if (!state.files.length) {
        dom.fileList.innerHTML = '<div class="empty">No logs found.</div>';
        return;
      }
      dom.fileList.innerHTML = state.files.map((file) => {
        const active = state.session && state.session.file === file.name;
        const title = file.scenario_title || "Unknown campaign";
        const sub = `${file.character_id || "?"} · ${file.turns_completed ?? "?"}/${file.max_turns ?? "?"} turns`;
        return `
          <button type="button" class="session-row ${active ? "active" : ""}" data-file="${escapeHtml(file.name)}" title="${escapeHtml(file.name)}" ${active ? 'aria-current="true"' : ""}>
            <span class="name">${escapeHtml(title)}</span>
            <span class="sub">${escapeHtml(sub)} · ${escapeHtml(formatDateTime(file.modified_time))}</span>
          </button>
        `;
      }).join("");
    }

    function renderTopbar() {
      const session = state.session;
      if (!session) return;
      const summary = session.summary;
      const title = summary.scenario_title || session.file;
      dom.sessionTitle.textContent = summary.character_id ? `${title} · ${summary.character_id}` : title;
      dom.sessionSub.textContent =
        `${summary.turns_completed ?? 0}/${summary.max_turns ?? "?"} turns · ` +
        `${session.stats.run_count} runs · ${formatDuration(summary.duration_s)}`;
    }

    function renderRailFilters() {
      const session = state.session;
      if (!session) return;

      const events = session.event_types.filter((event) => event !== "turn_started");
      dom.eventList.innerHTML = events.map((event) => `
        <button type="button" class="filter-row ${state.eventFilter === event ? "active" : ""}" data-event="${escapeHtml(event)}" aria-pressed="${state.eventFilter === event}">
          <span class="dot ${EVENT_COLOR[event] || ""}"></span>${escapeHtml(event.replaceAll("_", " "))}
          <span class="count">${escapeHtml(session.event_counts[event])}</span>
        </button>
      `).join("");

      const labels = [...new Set(session.runs.map((run) => run.label))];
      const labelCounts = {};
      session.entries.forEach((entry) => {
        if (entry.label) labelCounts[entry.label] = (labelCounts[entry.label] || 0) + 1;
      });
      dom.labelList.innerHTML = labels.length
        ? labels.map((label) => `
            <button type="button" class="filter-row ${state.labelFilter === label ? "active" : ""}" data-label="${escapeHtml(label)}" aria-pressed="${state.labelFilter === label}">
              <span class="dot violet"></span>${escapeHtml(label)}
              <span class="count">${escapeHtml(labelCounts[label] || 0)}</span>
            </button>
          `).join("")
        : '<div class="empty">No agent runs recorded.</div>';
    }

    function detailAllows(entry) {
      if (state.detail === "all") return true;
      if (state.detail === "runs") return entry.event !== "llm_message";
      return STORY_EVENTS.has(entry.event);
    }

    function filteredEntries() {
      if (!state.session) return [];
      const search = state.search.trim().toLowerCase();
      return state.session.entries.filter((entry) => {
        if (entry.event === "turn_started") return false;
        if (state.eventFilter ? entry.event !== state.eventFilter : !detailAllows(entry)) return false;
        if (state.labelFilter && entry.label !== state.labelFilter) return false;
        if (!search) return true;
        const haystack = [entry.summary, entry.excerpt, entry.label, entry.event, entry.raw_pretty]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        return haystack.includes(search);
      });
    }

    function rowContent(entry) {
      const raw = entry.raw || {};
      if (entry.event === "resolved" && raw.result) {
        const who = raw.subject ? `<span class="who">${escapeHtml(raw.subject)}</span> ` : "";
        return `${who}${escapeHtml(raw.result)}`;
      }
      if (entry.event === "world_snapshot") {
        const stats = [
          ["locations", raw.locations], ["characters", raw.characters],
          ["quests done", raw.quests_completed], ["xp", raw.total_xp], ["game min", raw.minutes_elapsed],
        ].filter(([, value]) => value !== undefined);
        return `<span class="stat-chips">${stats.map(([key, value]) => `<span>${escapeHtml(value)} ${escapeHtml(key)}</span>`).join("")}</span>`;
      }
      if (entry.event === "llm_message") {
        const tools = [...(entry.tool_calls || []), ...(entry.tool_returns || [])];
        if (tools.length) return `→ ${escapeHtml(tools.join(", "))}${entry.excerpt ? ` · ${escapeHtml(entry.excerpt)}` : ""}`;
        return escapeHtml(entry.excerpt || entry.summary);
      }
      return escapeHtml(entry.summary);
    }

    function rowChip(entry) {
      if (entry.event === "llm_message") {
        const kind = { model_request: "request", model_response: "response", tool_return: "tool ret" }[entry.message_kind] || "llm";
        return `<span class="chip">${escapeHtml(kind)}</span>`;
      }
      if (entry.event === "agent_run_started" || entry.event === "agent_run_finished") {
        const suffix = entry.event === "agent_run_finished" ? " ✓" : " …";
        return `<span class="chip violet">${escapeHtml((entry.label || "run") + suffix)}</span>`;
      }
      const label = { game_session_started: "session", game_session_finished: "session", campaign_completed: "victory", campaign_failed: "defeat", world_snapshot: "world", player_action: "player", pc_death: "death", resolved: (entry.raw || {}).tool || "resolved", parse_error: "error", run_error: "error", world_update_rejected: "rejected" }[entry.event] || entry.event.replaceAll("_", " ");
      return `<span class="chip ${EVENT_COLOR[entry.event] || ""}">${escapeHtml(label.toLowerCase())}</span>`;
    }

    function rowClass(entry) {
      if (entry.event === "resolved" || entry.event === "pc_death") return "story";
      if (entry.event === "llm_message" || entry.event === "agent_run_started" || entry.event === "agent_run_finished") return "dim";
      return "";
    }

    function renderRows() {
      const entries = filteredEntries();
      if (!entries.length) {
        dom.logPanel.innerHTML = '<div class="empty">No events match the current filters.</div>';
        return;
      }

      const chunks = [];
      let currentTurn;
      for (const entry of entries) {
        if (entry.turn !== currentTurn) {
          currentTurn = entry.turn;
          const title = currentTurn === null || currentTurn === undefined ? "Session" : `Turn ${currentTurn}`;
          chunks.push(`<div class="turn-header">${escapeHtml(title)}<span class="clock">${escapeHtml(entry.display_time)}</span></div>`);
        }
        chunks.push(`
          <button type="button" class="log-row ${rowClass(entry)} ${state.selectedLine === entry.line ? "selected" : ""}" data-line="${entry.line}" aria-pressed="${state.selectedLine === entry.line}" aria-label="Inspect log event on line ${entry.line}">
            <span class="log-time">${escapeHtml(entry.display_time)}</span>
            ${rowChip(entry)}
            <span class="log-msg">${rowContent(entry)}</span>
          </button>
        `);
      }
      dom.logPanel.innerHTML = chunks.join("");
    }

    function highlightPayload(text) {
      return escapeHtml(text)
        .replace(/^(ModelRequest|ModelResponse)(.*)$/m, '<span class="h">$1</span>$2')
        .replace(/^(\\s*)([\\w.-]+):/gm, '$1<span class="k">$2</span>:');
    }

    function renderInspector() {
      const session = state.session;
      if (!session) {
        dom.inspector.innerHTML = "";
        return;
      }
      const entry = session.entries.find((candidate) => candidate.line === state.selectedLine);
      if (!entry) {
        renderSessionInspector(session);
        return;
      }

      const tools = [...(entry.tool_calls || []), ...(entry.tool_returns || [])];
      const usage = entry.usage
        ? Object.entries(entry.usage).map(([key, value]) => `${key.replace("_tokens", "")} ${value.toLocaleString()}`).join(" · ")
        : null;

      dom.inspector.innerHTML = `
        <h4>Event Detail</h4>
        <div class="insp-field"><div class="insp-label">Summary</div><div class="insp-value">${escapeHtml(entry.summary)}</div></div>
        <div class="insp-field"><div class="insp-label">Event</div><div class="insp-value mono">${escapeHtml(entry.message_kind || entry.event)}</div></div>
        ${entry.label ? `<div class="insp-field"><div class="insp-label">Agent</div><div class="insp-value mono">${escapeHtml(entry.label)}</div></div>` : ""}
        <div class="insp-field"><div class="insp-label">Time</div><div class="insp-value mono">${escapeHtml(entry.display_time)} · turn ${escapeHtml(entry.turn ?? "-")} · line ${escapeHtml(entry.line)}</div></div>
        ${tools.length ? `<div class="insp-field"><div class="insp-label">Tools</div><div class="pill-row">${tools.map((tool) => `<span class="pill">${escapeHtml(tool)}</span>`).join("")}</div></div>` : ""}
        ${usage ? `<div class="insp-field"><div class="insp-label">Tokens</div><div class="insp-value mono">${escapeHtml(usage)}</div></div>` : ""}
        <div class="insp-field"><div class="insp-label">Payload</div><div class="payload">${highlightPayload(entry.raw_pretty || JSON.stringify(entry.raw, null, 2))}</div></div>
      `;
    }

    function renderSessionInspector(session) {
      const summary = session.summary;
      const tokens = { input_tokens: 0, output_tokens: 0 };
      session.runs.forEach((run) => {
        Object.keys(tokens).forEach((key) => { tokens[key] += run.tokens?.[key] || 0; });
      });

      dom.inspector.innerHTML = `
        <h4>Session Overview</h4>
        <div class="insp-field"><div class="insp-label">Campaign</div><div class="insp-value">${escapeHtml(summary.scenario_title || "unknown")}</div></div>
        <div class="insp-field"><div class="insp-label">Character</div><div class="insp-value mono">${escapeHtml(summary.character_id || "unknown")}</div></div>
        <div class="insp-field"><div class="insp-label">Started</div><div class="insp-value">${escapeHtml(formatDateTime(summary.started_at))}</div></div>
        <div class="insp-field"><div class="insp-label">Duration</div><div class="insp-value">${escapeHtml(formatDuration(summary.duration_s))} · ${escapeHtml(summary.turns_completed ?? 0)}/${escapeHtml(summary.max_turns ?? "?")} turns</div></div>
        <div class="insp-field"><div class="insp-label">Traffic</div><div class="insp-value">${escapeHtml(session.stats.run_count)} agent runs · ${escapeHtml(session.stats.llm_message_count)} LLM messages</div></div>
        <div class="insp-field"><div class="insp-label">Tokens</div><div class="insp-value mono">${escapeHtml(tokens.input_tokens.toLocaleString())} in · ${escapeHtml(tokens.output_tokens.toLocaleString())} out</div></div>
        ${session.stats.error_count ? `<div class="insp-field"><div class="insp-label">Errors</div><div class="insp-value">${escapeHtml(session.stats.error_count)}</div></div>` : ""}
        <div class="insp-field"><div class="insp-label">File</div><div class="insp-value mono">${escapeHtml(session.path)}</div></div>
        <div class="empty">Select a log row to inspect its payload.</div>
      `;
    }

    dom.fileList.addEventListener("click", (event) => {
      const button = event.target.closest("[data-file]");
      if (button) loadSession(button.dataset.file);
    });

    dom.eventList.addEventListener("click", (event) => {
      const button = event.target.closest("[data-event]");
      if (!button) return;
      state.eventFilter = state.eventFilter === button.dataset.event ? null : button.dataset.event;
      renderRailFilters();
      renderRows();
    });

    dom.labelList.addEventListener("click", (event) => {
      const button = event.target.closest("[data-label]");
      if (!button) return;
      state.labelFilter = state.labelFilter === button.dataset.label ? null : button.dataset.label;
      renderRailFilters();
      renderRows();
    });

    dom.detailSeg.addEventListener("click", (event) => {
      const button = event.target.closest("[data-detail]");
      if (!button) return;
      state.detail = button.dataset.detail;
      dom.detailSeg.querySelectorAll("button").forEach((b) => {
        const active = b === button;
        b.classList.toggle("active", active);
        b.setAttribute("aria-pressed", String(active));
      });
      renderRows();
    });

    dom.logPanel.addEventListener("click", (event) => {
      const row = event.target.closest("[data-line]");
      if (!row) return;
      state.selectedLine = state.selectedLine === Number(row.dataset.line) ? null : Number(row.dataset.line);
      renderRows();
      renderInspector();
      dom.logPanel.querySelector(`[data-line="${row.dataset.line}"]`)?.focus();
    });

    dom.search.addEventListener("input", (event) => {
      state.search = event.target.value;
      renderRows();
    });

    loadFiles().catch((error) => {
      dom.logPanel.innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
    });
