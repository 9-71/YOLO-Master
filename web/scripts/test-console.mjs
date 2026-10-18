import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const ts = require("typescript");
const root = path.resolve(import.meta.dirname, "../..");
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };
const flush = () => new Promise(setImmediate);
const response = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });

class Element {
  constructor() { this.children = []; this.value = ""; this.checked = true; this.style = {}; this.className = ""; this.disabled = false; this._text = ""; this.classList = { add() {}, remove() {}, toggle() {} }; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text; }
  get firstChild() { return this.children[0]; }
  appendChild(child) { this.children.push(child); return child; }
  append(...children) { this.children.push(...children); }
  removeChild(child) { this.children.splice(this.children.indexOf(child), 1); }
  addEventListener() {}
}

function consoleHarness(fetch, timing = {}) {
  const elements = new Map();
  const timers = new Map();
  let timer = 0;
  const context = vm.createContext({
    fetch, crypto: { randomUUID: () => "12345678-1234-1234-1234-123456789012" },
    localStorage: { getItem: () => null, setItem() {} },
    document: { getElementById(id) { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); }, createElement: () => new Element(), addEventListener() {} },
    window: { confirm: () => true, setTimeout(fn) { timers.set(++timer, fn); return timer; }, clearTimeout(id) { timers.delete(id); }, setInterval(fn) { timers.set(++timer, fn); return timer; }, clearInterval(id) { timers.delete(id); }, ...timing },
  });
  const source = fs.readFileSync(path.join(root, "frontend/app.js"), "utf8");
  vm.runInContext(source + "\nglobalThis.app = { state, selectJob, refreshDetail, pollLogs, loadArtifacts, logsTick, cancelJob, applyBaseUrl, onDispatchSubmit, renderArtifacts, extractDetail, jobDuration, refreshJobs, startJobsTimer, stopJobsTimer };", context);
  return { ...context.app, elements, timers };
}

// Execute production hooks/reducer with controlled effects and network responses.
// Browser layout/interaction acceptance remains a separate manual gate.
function reactHarness(fetch, timing = {}) {
  let state;
  const refs = [];
  let refIndex = 0;
  const effects = [];
  const timers = new Map();
  let timer = 0;
  const react = {
    createContext: () => ({ Provider: "provider" }), useContext: () => null,
    useCallback: (fn) => fn, useMemo: (fn) => fn(),
    useReducer(reducer, initial) { state ??= initial; return [state, (action) => { state = reducer(state, action); }]; },
    useRef(value) { const index = refIndex++; refs[index] ??= { current: value }; return refs[index]; },
    useEffect(fn, dependencies) { effects.push({ fn, dependencies }); },
  };
  const cache = new Map();
  const globals = { fetch, AbortController, localStorage: { getItem: () => null, setItem() {} }, window: { setTimeout(fn) { timers.set(++timer, fn); return timer; }, clearTimeout(id) { timers.delete(id); }, setInterval(fn) { timers.set(++timer, fn); return timer; }, clearInterval(id) { timers.delete(id); }, ...timing } };
  function load(filename) {
    if (cache.has(filename)) return cache.get(filename);
    const module = { exports: {} };
    cache.set(filename, module.exports);
    let source = fs.readFileSync(filename, "utf8").replace("import.meta.env.VITE_API_BASE", "undefined");
    if (filename.endsWith("JobContext.tsx")) source += "\nexport const testState = { reducer, initialState };";
    const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX } }).outputText;
    const localRequire = (name) => {
      if (name === "react") return react;
      if (name === "react/jsx-runtime") return { jsx: (type, props) => ({ type, props }) };
      if (name.startsWith(".")) { const base = path.resolve(path.dirname(filename), name); return load([base + ".ts", base + ".tsx"].find(fs.existsSync)); }
      return require(name);
    };
    vm.runInNewContext(compiled, { ...globals, module, exports: module.exports, require: localRequire }, { filename });
    return module.exports;
  }
  const context = load(path.join(root, "web/src/state/JobContext.tsx"));
  return { ...context.testState, effects, timers, dispatch(action) { state = context.testState.reducer(state ?? context.testState.initialState, action); }, render() { refIndex = 0; effects.length = 0; return context.JobProvider({ children: null }).props.value; } };
}

const detail = (jobId, status = "running") => ({ job_id: jobId, task_type: "diagnose", status, metadata: {}, artifact_count: 0, duration: 1 });

for (const status of ["completed", "failed", "cancelled"]) {
  test(`zero-build drains all terminal pages for ${status}`, async () => {
    const offsets = [];
    const app = consoleHarness(async (url) => {
      if (url.includes("/logs?")) { const offset = Number(new URL(url).searchParams.get("offset")); offsets.push(offset); const end = Math.min(offset + 500, 1601); return response({ offset, logs: Array.from({ length: end - offset }, (_, i) => String(offset + i)), next_offset: end < 1601 ? end : null }); }
      if (url.endsWith("/artifacts")) return response({ artifacts: [], image_artifacts: [] });
      return response(detail("one", status));
    });
    await app.selectJob("one");
    assert.deepEqual(offsets, [0, 500, 1000, 1500]);
    assert.equal(app.state.logOffset, 1601);
    assert.equal(app.elements.get("terminal-body").children.at(-1).textContent, "1600");
  });
}

test("zero-build ignores stale job and engine responses", async () => {
  const pending = deferred();
  const app = consoleHarness(async () => pending.promise);
  app.state.selectedJobId = "one";
  const request = app.refreshDetail();
  app.state.selectedJobId = "two";
  pending.resolve(response(detail("one")));
  await request;
  assert.equal(app.state.detail, null);
  const oldEngine = deferred();
  const other = consoleHarness(async () => oldEngine.promise);
  other.state.selectedJobId = "one";
  const oldRequest = other.loadArtifacts();
  other.state.baseUrl = "http://other:8000";
  oldEngine.resolve(response({ artifacts: [{ filename: "old.txt", download_url: "/old.txt" }], image_artifacts: [] }));
  await oldRequest;
  assert.equal(other.elements.has("artifact-files"), false);
});

test("zero-build Live off still observes terminal status and drains final logs", async () => {
  let logs = 0;
  const app = consoleHarness(async (url) => {
    if (url.includes("/logs?")) { logs++; return response({ offset: 0, logs: ["finished"], next_offset: null }); }
    if (url.endsWith("/artifacts")) return response({ artifacts: [], image_artifacts: [] });
    return response(detail("one", "completed"));
  });
  app.state.selectedJobId = "one";
  app.state.detail = detail("one");
  app.elements.set("live-logs", Object.assign(new Element(), { checked: false }));
  await app.logsTick();
  assert.equal(app.state.detail.status, "completed");
  assert.equal(logs, 1);
});

test("zero-build preserves nested artifact identity and deduplicates manifest images", () => {
  const app = consoleHarness(async () => response({}));
  app.state.selectedJobId = "one";
  app.renderArtifacts({ artifacts: [{ filename: "plot a.png", artifact_id: "plots/plot a.png", is_image: true, download_url: "/static/artifacts/one/plots/plot%20a.png" }], image_artifacts: ["plots/plot a.png"] });
  const gallery = app.elements.get("artifact-gallery");
  assert.equal(gallery.children.length, 1);
  assert.equal(gallery.children[0].children[0].src, "http://localhost:8000/static/artifacts/one/plots/plot%20a.png");
});

test("zero-build duration uses runtime execution duration", () => {
  const app = consoleHarness(async () => response({}));
  assert.equal(app.jobDuration({ job_id: "one", status: "completed", duration: 1.25, created_at: "2000-01-01" }), "1.25s");
});

test("React submit cannot navigate a job from the previous engine", async () => {
  const pending = deferred();
  const app = reactHarness(async (url, options) => options?.method === "POST" ? pending.promise : response({ jobs: [] }));
  const before = app.render();
  const submission = before.submitJob({ job_id: "old" });
  before.applyBaseUrl("http://other:8000");
  app.render();
  pending.resolve(response({ job_id: "old" }));
  assert.equal(await submission, null);
});

test("React reducer ignores stale detail/log/artifact identities and overlapping log offsets", () => {
  const { reducer, initialState } = reactHarness(async () => response({}));
  const selected = reducer(initialState, { type: "select", jobId: "one" });
  const guard = { rev: selected.baseRev, watchSeq: selected.watchSeq, jobId: "one" };
  const next = reducer(selected, { type: "select", jobId: "two" });
  for (const action of [{ type: "detail", value: detail("one") }, { type: "logs", lines: ["old"], offset: 1, expectedOffset: 0 }, { type: "artifacts", value: { job_id: "one", artifacts: [] } }]) assert.equal(reducer(next, { ...action, ...guard }), next);
  const logged = reducer(selected, { type: "logs", lines: ["one"], offset: 1, expectedOffset: 0, ...guard });
  assert.equal(reducer(logged, { type: "logs", lines: ["duplicate"], offset: 1, expectedOffset: 0, ...guard }), logged);
});

test("React cancel ACK remains pending until USER_CANCELLED terminal observation", () => {
  const { reducer, initialState } = reactHarness(async () => response({}));
  const pending = reducer(initialState, { type: "cancelStart", jobId: "one", opId: 1 });
  assert.equal(reducer(pending, { type: "cancelOutcome", jobId: "one", opId: 1, rev: 0, value: detail("one") }), pending);
  const cancelled = reducer(pending, { type: "cancelOutcome", jobId: "one", opId: 1, rev: 0, value: { ...detail("one", "cancelled"), error_code: "USER_CANCELLED" } });
  assert.equal(cancelled.cancellationPending, null);
  assert.match(cancelled.banner.message, /Cancellation completed/);
});


test("zero-build in-flight tail and terminal drain do not duplicate pages", async () => {
  const pending = deferred();
  const offsets = [];
  const app = consoleHarness(async (url) => {
    const offset = Number(new URL(url).searchParams.get("offset"));
    offsets.push(offset);
    if (offset === 0) return pending.promise;
    return response({ offset, logs: ["last"], next_offset: null });
  });
  app.state.selectedJobId = "one";
  const tail = app.pollLogs();
  const drain = app.pollLogs(true);
  pending.resolve(response({ offset: 0, logs: Array(500).fill("line"), next_offset: 500 }));
  await Promise.all([tail, drain]);
  assert.deepEqual(offsets, [0, 500]);
  assert.equal(app.state.logOffset, 501);
  assert.equal(app.elements.get("terminal-body").children.length, 501);
});

test("zero-build rejects stale logs after A to B to A selection", async () => {
  const pending = deferred();
  const app = consoleHarness(async () => pending.promise);
  app.state.selectedJobId = "one";
  const request = app.pollLogs();
  app.state.watchSeq += 2;
  pending.resolve(response({ offset: 0, logs: ["stale"], next_offset: null }));
  await request;
  assert.equal(app.state.logOffset, 0);
  assert.equal(app.elements.has("terminal-body"), false);
});

test("zero-build retries terminal log drain after a transient failure", async () => {
  let fail = true;
  const app = consoleHarness(async (url) => {
    if (url.includes("/logs?")) {
      if (fail) { fail = false; throw new Error("temporarily disconnected"); }
      return response({ offset: 0, logs: ["final"], next_offset: null });
    }
    if (url.endsWith("/artifacts")) return response({ artifacts: [], image_artifacts: [] });
    return response(detail("one", "completed"));
  });
  await app.selectJob("one");
  assert.notEqual(app.state.logsTimer, null);
  await app.logsTick();
  assert.equal(app.state.logsTimer, null);
  assert.equal(app.state.logOffset, 1);
});

test("zero-build cancellation ACK stays pending independently of selection", async () => {
  let status = "running";
  const app = consoleHarness(async (url) => {
    if (url.endsWith("/cancel")) return response({ status: "cancel_requested", message: "Accepted" }, 202);
    if (url.includes("?limit=")) return response({ jobs: [] });
    return response({ ...detail("one", status), error_code: status === "cancelled" ? "USER_CANCELLED" : null });
  });
  await app.cancelJob("one");
  assert.equal(app.state.cancellation.jobId, "one");
  assert.match(app.elements.get("banner-text").textContent, /Waiting for terminal/);
  app.state.selectedJobId = "another";
  status = "cancelled";
  app.timers.get(app.state.cancelTimer)();
  await flush();
  assert.equal(app.state.cancellation, null);
  assert.equal(app.state.selectedJobId, "another");
  assert.match(app.elements.get("banner-text").textContent, /Cancellation completed/);
});

test("zero-build ignores a submission ACK after an engine switch", async () => {
  const pending = deferred();
  const app = consoleHarness(async () => pending.promise);
  const values = { "task-type": "diagnose", "model-path": "", "data-source": "", device: "cpu", "output-dir": "runs/diagnose", "extra-params": "", "allowed-paths": "." };
  for (const [id, value] of Object.entries(values)) app.elements.set(id, Object.assign(new Element(), { value }));
  const request = app.onDispatchSubmit({ preventDefault() {} });
  app.state.baseRev += 1;
  pending.resolve(response({ job_id: "old" }, 201));
  await request;
  assert.equal(app.state.selectedJobId, null);
  assert.equal(app.elements.has("banner-text"), false);
});

test("structured Runtime admission failures retain code and message", () => {
  const app = consoleHarness(async () => response({}));
  assert.equal(app.extractDetail({ detail: { code: "PERSISTENCE_FAILED", message: "Snapshot unavailable" } }), "PERSISTENCE_FAILED: Snapshot unavailable");
});

function terminalEffect(app, status = "completed") {
  app.render().selectJob("one");
  const selected = app.render();
  app.dispatch({ type: "detail", rev: selected.baseRev, watchSeq: selected.watchSeq, jobId: "one", value: detail("one", status) });
  app.render();
  return app.effects.find(({ dependencies }) => dependencies?.[0] === status);
}

test("React terminal effect drains all pages with Live off", async () => {
  const offsets = [];
  const app = reactHarness(async (url) => {
    if (url.endsWith("/artifacts")) return response({ job_id: "one", artifacts: [], image_artifacts: [] });
    const offset = Number(new URL(url).searchParams.get("offset"));
    offsets.push(offset);
    const end = Math.min(offset + 500, 1601);
    return response({ offset, logs: Array.from({ length: end - offset }, (_, i) => String(offset + i)), next_offset: end < 1601 ? end : null });
  });
  app.render().setToggle("liveLogs", false);
  const effect = terminalEffect(app);
  const cleanup = effect.fn();
  await flush();
  const final = app.render();
  assert.deepEqual(offsets, [0, 500, 1000, 1500]);
  assert.equal(final.logOffset, 1601);
  assert.equal(final.logs.length, 1000);
  assert.equal(final.logs.at(-1), "1600");
  cleanup();
});

test("React retries a terminal drain after a transient failure", async () => {
  let fail = true;
  const app = reactHarness(async (url) => {
    if (url.endsWith("/artifacts")) return response({ job_id: "one", artifacts: [], image_artifacts: [] });
    if (fail) { fail = false; throw new Error("temporarily disconnected"); }
    return response({ offset: 0, logs: ["final"], next_offset: null });
  });
  const cleanup = terminalEffect(app).fn();
  await flush();
  assert.equal(app.timers.size, 1);
  await [...app.timers.values()][0]();
  assert.equal(app.render().logOffset, 1);
  cleanup();
});

test("React stops old terminal pagination immediately after an engine switch", async () => {
  const second = deferred();
  const requested = deferred();
  const urls = [];
  const app = reactHarness(async (url) => {
    if (url.endsWith("/artifacts")) return response({ job_id: "one", artifacts: [], image_artifacts: [] });
    urls.push(url);
    if (urls.length === 2) { requested.resolve(); return second.promise; }
    return response({ offset: 0, logs: Array(500).fill("line"), next_offset: 500 });
  });
  const cleanup = terminalEffect(app).fn();
  await requested.promise;
  app.render().applyBaseUrl("http://other:8000");
  app.render();
  second.resolve(response({ offset: 500, logs: Array(500).fill("old"), next_offset: 1000 }));
  await flush();
  assert.equal(urls.length, 2);
  assert.ok(urls.every((url) => url.startsWith("http://localhost:8000/")));
  assert.equal(app.render().logOffset, 0);
  cleanup();
});

test("React stale watch errors do not replace the current banner", () => {
  const { reducer, initialState } = reactHarness(async () => response({}));
  const one = reducer(initialState, { type: "select", jobId: "one" });
  const two = reducer(one, { type: "select", jobId: "two" });
  assert.equal(reducer(two, { type: "banner", rev: 0, watchSeq: one.watchSeq, jobId: "one", value: { kind: "error", message: "old" } }), two);
});


test("React retries missing initial status after a transient detail failure", async () => {
  let attempts = 0;
  const app = reactHarness(async (url) => {
    if (url.endsWith("/artifacts")) return response({ job_id: "one", artifacts: [], image_artifacts: [] });
    if (url.includes("/logs?")) return response({ offset: 0, logs: [], next_offset: null });
    if (++attempts === 1) throw new Error("temporarily disconnected");
    return response(detail("one"));
  });
  app.render().selectJob("one");
  app.render();
  const snapshot = app.effects.find(({ dependencies }) => dependencies?.length === 3);
  const cleanupSnapshot = snapshot.fn();
  await flush();
  assert.equal(app.render().detail, null);
  const monitoring = app.effects.find(({ dependencies }) => dependencies?.length === 6 && typeof dependencies[0] === "function");
  const cleanupMonitoring = monitoring.fn();
  assert.equal(app.timers.size, 1);
  [...app.timers.values()][0]();
  await flush();
  assert.equal(app.render().detail.status, "running");
  cleanupSnapshot();
  cleanupMonitoring();
});

test("React terminal artifact fetch retries after a transient failure", async () => {
  let attempts = 0;
  const app = reactHarness(async (url) => {
    if (url.endsWith("/artifacts")) {
      if (++attempts === 1) throw new Error("temporarily disconnected");
      return response({ job_id: "one", artifacts: [{ filename: "result.json", artifact_id: "result.json", is_image: false, download_url: "/static/artifacts/one/result.json" }], image_artifacts: [] });
    }
    return response({ offset: 0, logs: [], next_offset: null });
  });
  const cleanup = terminalEffect(app).fn();
  await flush();
  assert.equal(app.timers.size, 1);
  await [...app.timers.values()][0]();
  assert.equal(app.render().artifacts.artifacts[0].filename, "result.json");
  cleanup();
});


test("React ignores an older detail failure after a newer successful status", async () => {
  const first = deferred();
  let attempts = 0;
  const app = reactHarness(async (url) => {
    if (url.endsWith("/artifacts")) return response({ job_id: "one", artifacts: [], image_artifacts: [] });
    if (url.includes("/logs?")) return response({ offset: 0, logs: [], next_offset: null });
    if (++attempts === 1) return first.promise;
    return response(detail("one"));
  });
  app.render().selectJob("one");
  app.render();
  const cleanupSnapshot = app.effects.find(({ dependencies }) => dependencies?.length === 3).fn();
  const cleanupMonitoring = app.effects.find(({ dependencies }) => dependencies?.length === 6 && typeof dependencies[0] === "function").fn();
  [...app.timers.values()][0]();
  await flush();
  first.resolve(response({ detail: "old response" }, 404));
  await flush();
  const current = app.render();
  assert.equal(current.detail.status, "running");
  assert.equal(current.detailMissing, false);
  cleanupSnapshot();
  cleanupMonitoring();
});


for (const olderFailure of [false, true]) {
  test(`React keeps terminal artifacts when an older request ${olderFailure ? "fails" : "returns empty"}`, async () => {
    const first = deferred();
    let attempts = 0;
    const manifest = { job_id: "one", artifacts: [{ filename: "result.json", artifact_id: "result.json", is_image: false, download_url: "/static/artifacts/one/result.json" }], image_artifacts: [] };
    const app = reactHarness(async (url) => {
      if (url.endsWith("/artifacts")) return ++attempts === 1 ? first.promise : response(manifest);
      return response({ offset: 0, logs: [], next_offset: null });
    });
    app.render().selectJob("one");
    const initialRequest = app.render().refreshArtifacts(true);
    const cleanup = terminalEffect(app).fn();
    await flush();
    assert.equal(app.render().artifacts.artifacts.length, 1);
    first.resolve(olderFailure ? response({ detail: "old failure" }, 503) : response({ job_id: "one", artifacts: [], image_artifacts: [] }));
    await initialRequest;
    const current = app.render();
    assert.equal(current.artifacts.artifacts.length, 1);
    assert.equal(current.banner, null);
    cleanup();
  });

  test(`zero-build keeps newer same-job artifacts when an older request ${olderFailure ? "fails" : "returns empty"}`, async () => {
    const first = deferred();
    let attempts = 0;
    const app = consoleHarness(async () => ++attempts === 1 ? first.promise : response({ artifacts: [{ filename: "result.json", artifact_id: "result.json", is_image: false, download_url: "/static/artifacts/one/result.json" }], image_artifacts: [] }));
    app.state.selectedJobId = "one";
    const initialRequest = app.loadArtifacts();
    await app.loadArtifacts();
    assert.equal(app.elements.get("artifact-files").children.length, 1);
    first.resolve(olderFailure ? response({ detail: "old failure" }, 503) : response({ artifacts: [], image_artifacts: [] }));
    await initialRequest;
    assert.equal(app.elements.get("artifact-files").children.length, 1);
  });
}

// Virtual time drives the actual production timers; no wall-clock latency.
function fakeClock() {
  let now = 0;
  let nextId = 0;
  const pending = new Map();
  const schedule = (fn, ms, repeat) => {
    const id = ++nextId;
    pending.set(id, { fn, at: now + ms, repeat });
    return id;
  };
  return {
    setTimeout: (fn, ms) => schedule(fn, ms, 0),
    clearTimeout: (id) => pending.delete(id),
    setInterval: (fn, ms) => schedule(fn, ms, ms),
    clearInterval: (id) => pending.delete(id),
    async advance(ms) {
      const end = now + ms;
      for (;;) {
        const next = [...pending.entries()].sort((a, b) => a[1].at - b[1].at || a[0] - b[0])[0];
        if (!next || next[1].at > end) break;
        const [id, entry] = next;
        now = entry.at;
        if (entry.repeat) entry.at += entry.repeat;
        else pending.delete(id);
        entry.fn();
        await flush();
      }
      now = end;
      await flush();
    },
  };
}

function startDetailEffects(app) {
  const snapshot = app.effects.find(({ dependencies }) => dependencies?.length === 3).fn();
  const monitoring = app.effects.find(({ dependencies }) => dependencies?.length === 6 && typeof dependencies[0] === "function").fn();
  return { snapshot, monitoring };
}

for (const initialStatus of [null, "running"]) {
  test(`React sustained 1500ms detail responses progress from ${initialStatus} and trigger terminal log drain`, async () => {
    const clock = fakeClock();
    let attempts = 0;
    const offsets = [];
    const app = reactHarness(async (url) => {
      if (url.endsWith("/artifacts")) return response({ job_id: "one", artifacts: [], image_artifacts: [] });
      if (url.includes("/logs?")) {
        const offset = Number(new URL(url).searchParams.get("offset"));
        offsets.push(offset);
        const end = Math.min(offset + 500, 1601);
        return response({ offset, logs: Array.from({ length: end - offset }, (_, i) => String(offset + i)), next_offset: end < 1601 ? end : null });
      }
      const attempt = ++attempts;
      return new Promise((resolve) => clock.setTimeout(() => resolve(response({ ...detail("one", attempt < 4 ? "running" : "completed"), metadata: { attempt } })), 1500));
    }, clock);
    app.render().selectJob("one");
    const selected = app.render();
    if (initialStatus) {
      app.dispatch({ type: "detail", rev: selected.baseRev, watchSeq: selected.watchSeq, jobId: "one", value: detail("one", initialStatus) });
      app.render();
    }
    app.render().setToggle("liveLogs", false);
    app.render();
    const cleanup = startDetailEffects(app);
    for (let attempt = 1; attempt <= 4; attempt++) {
      await clock.advance(attempt === 1 ? 1500 : 1000);
      const current = app.render();
      assert.equal(current.detail?.metadata.attempt, attempt);
      assert.equal(current.detail.status, attempt < 4 ? "running" : "completed");
      assert.equal(current.detailMissing, false);
    }
    assert.equal(attempts, 5); // A newer request was pending at every response.
    cleanup.monitoring(); // React stops monitoring after observing terminal.
    const stopDrain = app.effects.find(({ dependencies }) => dependencies?.[0] === "completed").fn();
    await flush();
    const final = app.render();
    assert.deepEqual(offsets, [0, 500, 1000, 1500]);
    assert.equal(final.logOffset, 1601);
    assert.equal(final.logs.length, 1000);
    assert.equal(final.logs.at(-1), "1600");
    stopDrain();
    cleanup.snapshot();
    await clock.advance(1500); // Remaining detail request is invalid after unmount.
    assert.equal(app.render().detail.metadata.attempt, 4);
  });
}

test("zero-build sustained 2500ms nonempty lists render despite 2000ms polling", async () => {
  const clock = fakeClock();
  let attempts = 0;
  const app = consoleHarness(async () => {
    const attempt = ++attempts;
    return new Promise((resolve) => clock.setTimeout(() => resolve(response({ jobs: [detail(`job-${attempt}`, "completed")] })), 2500));
  }, clock);
  void app.refreshJobs();
  app.startJobsTimer();
  for (let attempt = 1; attempt <= 5; attempt++) {
    await clock.advance(attempt === 1 ? 2500 : 2000);
    assert.equal(app.state.jobs[0]?.job_id, `job-${attempt}`);
    const rows = app.elements.get("jobs-body").children;
    assert.equal(rows.length, 1);
    assert.equal(rows[0].children[0].textContent, `job-${attempt}`);
    assert.equal(app.elements.get("conn-text").textContent, "Connected");
  }
  assert.equal(attempts, 6);
  app.stopJobsTimer();
  await clock.advance(2500);
});

for (const olderFailure of [false, true]) {
  test(`zero-build newer list survives older ${olderFailure ? "failure" : "success"}`, async () => {
    const first = deferred();
    let attempts = 0;
    const app = consoleHarness(async () => ++attempts === 1 ? first.promise : response({ jobs: [detail("new", "completed")] }));
    const oldRequest = app.refreshJobs();
    await app.refreshJobs();
    first.resolve(olderFailure ? response({ detail: "old failure" }, 503) : response({ jobs: [detail("old")] }));
    await oldRequest;
    assert.equal(app.state.jobs[0].job_id, "new");
    assert.equal(app.elements.get("conn-text").textContent, "Connected");
  });

  test(`React newer detail survives older ${olderFailure ? "failure" : "success"}`, async () => {
    const first = deferred();
    let attempts = 0;
    const app = reactHarness(async (url) => {
      if (url.endsWith("/artifacts")) return response({ job_id: "one", artifacts: [], image_artifacts: [] });
      if (url.includes("/logs?")) return response({ offset: 0, logs: [], next_offset: null });
      return ++attempts === 1 ? first.promise : response(detail("one", "completed"));
    });
    app.render().selectJob("one");
    app.render();
    const cleanup = startDetailEffects(app);
    [...app.timers.values()][0]();
    await flush();
    first.resolve(olderFailure ? response({ detail: "old missing" }, 404) : response({ ...detail("one"), error_code: "SEC_ERR_001" }));
    await flush();
    const current = app.render();
    assert.equal(current.detail.status, "completed");
    assert.equal(current.detailMissing, false);
    assert.equal(current.banner, null);
    cleanup.monitoring();
    cleanup.snapshot();
  });
}

for (const boundary of ["selection", "engine", "unmount"]) {
  for (const failure of [false, true]) {
    test(`React discards pending detail ${failure ? "failure" : "success"} across ${boundary}`, async () => {
      const pending = deferred();
      const app = reactHarness(async (url) => {
        if (url.endsWith("/artifacts")) return response({ job_id: "one", artifacts: [], image_artifacts: [] });
        if (url.includes("/logs?")) return response({ offset: 0, logs: [], next_offset: null });
        return pending.promise;
      });
      app.render().selectJob("one");
      app.render();
      const cleanup = startDetailEffects(app);
      await flush();
      if (boundary === "selection") { app.render().selectJob("two"); app.render().selectJob("one"); }
      if (boundary === "engine") { app.render().applyBaseUrl("http://other:8000"); app.render().applyBaseUrl("http://localhost:8000"); app.render().selectJob("one"); }
      cleanup.monitoring();
      cleanup.snapshot();
      pending.resolve(failure ? response({ detail: "old missing" }, 404) : response({ ...detail("one", "completed"), error_code: "SEC_ERR_001" }));
      await flush();
      const current = app.render();
      assert.equal(current.detail, null);
      assert.equal(current.detailMissing, false);
      assert.equal(current.banner, null);
    });
  }
}

test("zero-build rejects pending list after engine A to B to A and recovers from error", async () => {
  const old = deferred();
  const clock = fakeClock();
  let attempts = 0;
  const app = consoleHarness(async () => {
    const attempt = ++attempts;
    if (attempt === 1) return old.promise;
    return new Promise((resolve) => clock.setTimeout(() => resolve(attempt === 2 ? response({ detail: "unavailable" }, 503) : response({ jobs: [detail("current", "completed")] })), 2500));
  }, clock);
  void app.refreshJobs();
  app.state.baseUrl = "http://other:8000";
  app.state.baseRev++;
  app.state.baseUrl = "http://localhost:8000";
  app.state.baseRev++;
  void app.refreshJobs();
  app.startJobsTimer();
  await clock.advance(2500);
  assert.equal(app.elements.get("conn-text").textContent, "Offline");
  assert.equal(app.state.jobs.length, 0);
  old.resolve(response({ jobs: [detail("stale")] }));
  await flush();
  assert.equal(app.state.jobs.length, 0);
  await clock.advance(2000);
  assert.equal(app.state.jobs[0].job_id, "current");
  assert.equal(app.elements.get("conn-text").textContent, "Connected");
  app.stopJobsTimer();
  await clock.advance(2500);
});

for (const failure of [false, true]) {
  test(`React same-identity effect restart accepts fresh detail and ignores old ${failure ? "failure" : "success"}`, async () => {
    const first = deferred();
    let attempts = 0;
    const app = reactHarness(async (url) => {
      if (url.endsWith("/artifacts")) return response({ job_id: "one", artifacts: [], image_artifacts: [] });
      if (url.includes("/logs?")) return response({ offset: 0, logs: [], next_offset: null });
      return ++attempts === 1 ? first.promise : response(detail("one"));
    });
    app.render().selectJob("one");
    app.render();
    const old = startDetailEffects(app);
    old.monitoring();
    old.snapshot();
    const current = startDetailEffects(app); // StrictMode setup -> cleanup -> setup.
    await flush();
    first.resolve(failure ? response({ detail: "old missing" }, 404) : response({ ...detail("one", "failed"), error_code: "SEC_ERR_001" }));
    await flush();
    assert.equal(app.render().detail.status, "running");
    assert.equal(app.render().detailMissing, false);
    assert.equal(app.render().banner, null);
    current.monitoring();
    current.snapshot();
  });
}

test("React slow initial failure is reported while a newer poll is pending and then recovers", async () => {
  const clock = fakeClock();
  let attempts = 0;
  const app = reactHarness(async (url) => {
    if (url.endsWith("/artifacts")) return response({ job_id: "one", artifacts: [], image_artifacts: [] });
    if (url.includes("/logs?")) return response({ offset: 0, logs: [], next_offset: null });
    const attempt = ++attempts;
    return new Promise((resolve) => clock.setTimeout(() => resolve(attempt === 1 ? response({ detail: "temporary failure" }, 503) : response(detail("one"))), 1500));
  }, clock);
  app.render().selectJob("one");
  app.render();
  const cleanup = startDetailEffects(app);
  await clock.advance(1500);
  assert.match(app.render().banner?.message ?? "", /503/);
  assert.equal(app.render().detail, null);
  await clock.advance(1000);
  assert.equal(app.render().detail.status, "running");
  assert.equal(app.render().detailMissing, false);
  cleanup.monitoring();
  cleanup.snapshot();
  await clock.advance(1500);
});

test("zero-build newer failure supersedes an older successful list and allows recovery", async () => {
  const first = deferred();
  let attempts = 0;
  const app = consoleHarness(async () => {
    if (++attempts === 1) return first.promise;
    return attempts === 2 ? response({ detail: "new failure" }, 503) : response({ jobs: [detail("recovered", "completed")] });
  });
  const old = app.refreshJobs();
  await app.refreshJobs();
  first.resolve(response({ jobs: [detail("old")] }));
  await old;
  assert.equal(app.state.jobs.length, 0);
  assert.equal(app.elements.get("conn-text").textContent, "Offline");
  await app.refreshJobs();
  assert.equal(app.state.jobs[0].job_id, "recovered");
  assert.equal(app.elements.get("conn-text").textContent, "Connected");
});

test("React overlapping 404 cannot overwrite a newer terminal detail between microtasks", async () => {
  for (let steps = 0; steps < 14; steps++) {
    const first = deferred();
    const second = deferred();
    let attempts = 0;
    const app = reactHarness(async (url) => {
      if (url.endsWith("/artifacts")) return response({ job_id: "one", artifacts: [], image_artifacts: [] });
      if (url.includes("/logs?")) return response({ offset: 0, logs: [], next_offset: null });
      return ++attempts === 1 ? first.promise : second.promise;
    });
    app.render().selectJob("one");
    app.render();
    const cleanup = startDetailEffects(app);
    [...app.timers.values()][0]();
    first.resolve(response({ detail: "old missing" }, 404));
    for (let turn = 0; turn < steps; turn++) await Promise.resolve();
    second.resolve(response(detail("one", "completed")));
    await flush();
    const current = app.render();
    assert.equal(current.detailMissing, false, `microtask gap ${steps}`);
    assert.equal(current.detail?.status, "completed", `microtask gap ${steps}`);
    cleanup.monitoring();
    cleanup.snapshot();
  }
});
