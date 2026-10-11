const state = {
  config: null,
  sessionId: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function adminHeaders() {
  return {
    "content-type": "application/json",
    "x-admin-token": $("#adminToken").value,
  };
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : text || response.statusText;
    throw new Error(detail);
  }
  return body;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function parseJsonField(element, fallback) {
  const raw = element.value.trim();
  if (!raw) return fallback;
  return JSON.parse(raw);
}

function lineList(value) {
  return (value || []).join("\n");
}

function fromLineList(value) {
  return value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function setStatus(health) {
  const line = $("#statusLine");
  const bits = [
    `status=${health.status}`,
    `llm=${health.llm_configured ? "ready" : "missing"}`,
    `quran=${health.quran_db_ready ? "ready" : "missing"}`,
    `hadith=${health.hadith_db_ready ? "ready" : "missing"}`,
    `mcp=${health.enabled_mcp_servers}`,
  ];
  line.textContent = bits.join("  ");
  line.className = health.status === "ok" ? "status-ok" : "status-warn";
}

async function refreshHealth() {
  try {
    setStatus(await requestJson("/health"));
  } catch (error) {
    $("#statusLine").textContent = error.message;
    $("#statusLine").className = "status-error";
  }
}

function renderLlm(config) {
  $("#llmProvider").value = config.llm.provider;
  $("#llmModel").value = config.llm.model;
  $("#llmBaseUrl").value = config.llm.base_url || "";
  $("#llmApiKey").value = config.llm.api_key || "";
  $("#llmOrganization").value = config.llm.organization || "";
  $("#llmProject").value = config.llm.project || "";
  $("#llmUseResponses").checked = config.llm.use_responses !== false;
  $("#agentTimeout").value = config.agent_timeout_seconds;
  $("#agentMaxTurns").value = config.agent_max_turns;
  $("#sessionTurns").value = config.session_max_turns;
}

function mcpNode(server) {
  const node = $("#mcpTemplate").content.firstElementChild.cloneNode(true);
  $(".mcp-enabled", node).checked = server.enabled;
  $(".mcp-id", node).value = server.id;
  $(".mcp-name", node).value = server.name;
  $(".mcp-transport", node).value = server.transport;
  $(".mcp-command", node).value = server.command || "";
  $(".mcp-args", node).value = lineList(server.args);
  $(".mcp-cwd", node).value = server.cwd || "";
  $(".mcp-url", node).value = server.url || "";
  $(".mcp-timeout", node).value = server.timeout_seconds;
  $(".mcp-sse-timeout", node).value = server.sse_read_timeout_seconds;
  $(".mcp-approval", node).value = server.require_approval;
  $(".mcp-structured", node).checked = server.use_structured_content;
  $(".mcp-env", node).value = pretty(server.env || {});
  $(".mcp-headers", node).value = pretty(server.headers || {});
  $(".remove-server", node).addEventListener("click", () => node.remove());
  return node;
}

function renderMcp(config) {
  const list = $("#mcpList");
  list.innerHTML = "";
  for (const server of config.mcp_servers) {
    list.appendChild(mcpNode(server));
  }
}

function gatherConfig() {
  if (!state.config) {
    throw new Error("Config belum dimuat. Isi admin token lalu klik Reload di tab Config.");
  }
  const config = structuredClone(state.config);
  config.llm = {
    provider: $("#llmProvider").value,
    model: $("#llmModel").value.trim(),
    base_url: $("#llmBaseUrl").value.trim() || null,
    api_key: $("#llmApiKey").value.trim() || null,
    organization: $("#llmOrganization").value.trim() || null,
    project: $("#llmProject").value.trim() || null,
    use_responses: $("#llmUseResponses").checked,
  };
  config.agent_timeout_seconds = Number($("#agentTimeout").value);
  config.agent_max_turns = Number($("#agentMaxTurns").value);
  config.session_max_turns = Number($("#sessionTurns").value);
  config.mcp_servers = $$(".server").map((node) => ({
    id: $(".mcp-id", node).value.trim(),
    name: $(".mcp-name", node).value.trim(),
    enabled: $(".mcp-enabled", node).checked,
    transport: $(".mcp-transport", node).value,
    command: $(".mcp-command", node).value.trim() || null,
    args: fromLineList($(".mcp-args", node).value),
    cwd: $(".mcp-cwd", node).value.trim() || null,
    url: $(".mcp-url", node).value.trim() || null,
    env: parseJsonField($(".mcp-env", node), {}),
    headers: parseJsonField($(".mcp-headers", node), {}),
    timeout_seconds: Number($(".mcp-timeout", node).value),
    sse_read_timeout_seconds: Number($(".mcp-sse-timeout", node).value),
    use_structured_content: $(".mcp-structured", node).checked,
    require_approval: $(".mcp-approval", node).value,
  }));
  return config;
}

async function loadConfig() {
  const body = await requestJson("/api/config", { headers: adminHeaders() });
  state.config = body.config;
  renderLlm(state.config);
  renderMcp(state.config);
  return body;
}

async function saveConfig() {
  const body = await requestJson("/api/config", {
    method: "PUT",
    headers: adminHeaders(),
    body: JSON.stringify(gatherConfig()),
  });
  state.config = body.config;
  renderLlm(state.config);
  renderMcp(state.config);
  await refreshHealth();
  return body;
}

function addServer() {
  const id = `mcp-${Date.now()}`;
  $("#mcpList").appendChild(
    mcpNode({
      id,
      name: id,
      enabled: true,
      transport: "streamable_http",
      command: null,
      args: [],
      cwd: null,
      url: "https://example.com/mcp",
      env: {},
      headers: {},
      timeout_seconds: 45,
      sse_read_timeout_seconds: 300,
      use_structured_content: true,
      require_approval: "never",
    }),
  );
}

async function testMcp() {
  const output = $("#mcpOutput");
  output.textContent = "Testing MCP servers...";
  if (!state.config) {
    await loadConfig();
  }
  await saveConfig();
  const body = await requestJson("/api/mcp/test", {
    method: "POST",
    headers: adminHeaders(),
  });
  output.textContent = pretty(body);
}

async function sendChat() {
  const output = $("#chatOutput");
  const status = $("#chatStatus");
  output.textContent = "";
  status.textContent = "Menghubungi bot…";
  let response;
  try {
    response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        message: $("#chatInput").value,
        session_id: state.sessionId,
        language: "id",
      }),
    });
  } catch (error) {
    status.textContent = "";
    output.textContent = error.message;
    return;
  }
  if (!response.ok || !response.body) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      /* keep status text */
    }
    status.textContent = "";
    output.textContent = `Error ${response.status}: ${detail}`;
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let index;
      while ((index = buffer.indexOf("\n\n")) >= 0) {
        const raw = buffer.slice(0, index);
        buffer = buffer.slice(index + 2);
        let event = "message";
        const dataParts = [];
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) dataParts.push(line.slice(5).trim());
        }
        if (!dataParts.length) continue;
        let payload = {};
        try {
          payload = JSON.parse(dataParts.join("\n"));
        } catch {
          continue;
        }
        if (event === "delta") {
          answer += payload.text || "";
          output.textContent = answer;
          status.textContent = "✍ Menyusun jawaban…";
        } else if (event === "tool") {
          status.textContent = `🔍 ${payload.name || "tool"}…`;
        } else if (event === "done") {
          state.sessionId = payload.session_id;
          output.textContent = payload.answer + "\n\n" + pretty({ references: payload.references, used_tools: payload.used_tools });
          const warnings = payload.warnings || [];
          status.textContent = warnings.length ? `⚠ ${warnings.join("; ")}` : "Selesai ✓";
        } else if (event === "error") {
          status.textContent = "";
          output.textContent = `Error: ${payload.detail || "unknown"}`;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function switchView(viewId) {
  $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === viewId));
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === viewId));
}

function wireEvents() {
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => switchView(tab.dataset.view)));
  $("#rememberToken").addEventListener("click", () => {
    localStorage.setItem("muslim_bot_admin_token", $("#adminToken").value);
    loadConfig().catch((error) => alert(error.message));
  });
  $("#reloadConfig").addEventListener("click", () => loadConfig().catch((error) => alert(error.message)));
  $("#saveConfig").addEventListener("click", () => saveConfig().catch((error) => alert(error.message)));
  $("#addMcp").addEventListener("click", addServer);
  $("#testMcp").addEventListener("click", () => testMcp().catch((error) => ($("#mcpOutput").textContent = error.message)));
  $("#sendChat").addEventListener("click", () => sendChat().catch((error) => ($("#chatOutput").textContent = error.message)));
  $("#copyDeploy").addEventListener("click", () => navigator.clipboard.writeText($("#deployCommands").textContent));
}

async function init() {
  $("#adminToken").value = localStorage.getItem("muslim_bot_admin_token") || "";
  wireEvents();
  await refreshHealth();
  if ($("#adminToken").value) {
    loadConfig().catch((error) => {
      $("#mcpOutput").textContent = error.message;
    });
  }
}

init();
