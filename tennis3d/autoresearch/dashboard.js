"use strict";
const $ = id => document.getElementById(id);
let state = null, selected = null, details = null, connected = false, fetchedAt = 0;
let comparison = null, comparisonError = "", detailError = "", detailRequest = 0;
let pendingContext = null;
const detailContext = () => JSON.stringify([$("segment").value, selected, $("compare-best").checked, state?.best[$("segment").value]?.trial]);
const curvePoints = data => (data?.curve || []).filter(p => Number.isFinite(p.loss) && Number.isFinite(p.train_seconds)).map(p => [p.train_seconds, p.loss]);
const number = (value, digits = 3) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
const text = (id, value) => { $(id).textContent = value; };
const trials = () => state ? state.trials.filter(t => t.segment === $("segment").value) : [];
const isThroughput = () => state?.segments.find(s => s.segment === $("segment").value)?.kind === "throughput";
const chartHovers = new Map();
const nearest = (samples, value) => samples.reduce((best, sample) => !best || Math.abs(sample[0] - value) < Math.abs(best[0] - value) ? sample : best, null);

function chartHover(canvas, plot) {
  let hover = chartHovers.get(canvas);
  if (!hover) {
    const tooltip = document.createElement("div");
    tooltip.className = "chart-tooltip"; tooltip.id = `${canvas.id}-tooltip`; tooltip.hidden = true;
    tooltip.setAttribute("role", "tooltip"); document.body.append(tooltip);
    hover = {tooltip, pointer:null, plot:null}; chartHovers.set(canvas, hover);
    hover.clear = () => { hover.pointer = null; tooltip.hidden = true; };
    hover.show = () => {
      const {left, right, top, xmax, at} = hover.plot;
      const bounds = canvas.getBoundingClientRect(), {clientX, clientY} = hover.pointer;
      const px = (clientX - bounds.left) * canvas.clientWidth / bounds.width;
      const py = (clientY - bounds.top) * canvas.clientHeight / bounds.height;
      if (px < left || px > right || py < top || py > canvas.clientHeight) { hover.clear(); return; }
      const hit = at((px - left) / (right - left) * xmax);
      if (!hit) { hover.clear(); return; }
      tooltip.textContent = hit.lines.join("\n"); tooltip.hidden = false;
      const size = tooltip.getBoundingClientRect();
      tooltip.style.left = `${Math.max(8, Math.min(clientX + 14, window.innerWidth - size.width - 8))}px`;
      tooltip.style.top = `${Math.max(8, Math.min(clientY + 14, window.innerHeight - size.height - 8))}px`;
    };
    canvas.addEventListener("pointermove", event => { hover.pointer = {clientX:event.clientX, clientY:event.clientY}; hover.show(); });
    canvas.addEventListener("pointerleave", hover.clear);
    canvas.addEventListener("pointercancel", hover.clear);
  }
  if (hover.plot?.context !== plot.context) { hover.pointer = null; hover.tooltip.hidden = true; }
  hover.plot = plot;
  if (hover.pointer) hover.show();
}
const clearChartHovers = () => chartHovers.forEach(hover => hover.clear());

function chart(id, series, xLabel, yLabel, failures = [], attemptAxis = false, hover = {}) {
  const canvas = $(id), scale = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (!width || !height) { chartHovers.get(canvas)?.clear(); return; }
  canvas.width = width * scale; canvas.height = height * scale;
  const ctx = canvas.getContext("2d"); ctx.scale(scale, scale);
  const left = 72, top = 20, right = width - 20, bottom = height - (attemptAxis ? 72 : 40);
  const finite = series.flatMap(s => s.points).filter(p => p.every(Number.isFinite));
  const points = finite.filter(p => p[1] > 0), logs = points.map(p => Math.log10(p[1]));
  const xmax = Math.max(1, ...finite.map(p => p[0]), ...failures, hover.xmax || 0);
  const minimum = points.length ? Math.min(...logs) : 0, maximum = points.length ? Math.max(...logs) : 1;
  const padding = points.length ? Math.max(0.05, (maximum - minimum) * 0.08) : 0;
  const ymin = minimum - padding, ymax = maximum + padding;
  const x = value => left + value / xmax * (right - left);
  const y = value => bottom - (Math.log10(value)-ymin) / (ymax-ymin) * (bottom - top);
  ctx.font = "12px system-ui"; ctx.fillStyle = "#546273"; ctx.strokeStyle = "#d6dee7";
  for (let i = 0; i <= 4; i++) {
    const exponent = ymin + (ymax-ymin) * i / 4, value = 10 ** exponent, py = bottom - i / 4 * (bottom - top);
    ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
    const power = Math.floor(exponent);
    const label = value > 0 && Number.isFinite(value)
      ? value < 0.01 || value >= 1000 ? value.toExponential(1) : String(Number(value.toPrecision(3)))
      : `${(10 ** (exponent - power)).toFixed(1)}e${power >= 0 ? "+" : ""}${power}`;
    ctx.fillText(label, 4, py + 4);
  }
  const step = Math.max(1, Math.ceil(xmax / 4));
  const ticks = attemptAxis ? [...new Set([...Array.from({length: Math.floor(xmax / step) + 1}, (_, i) => i * step), xmax])] : Array.from({length:5}, (_, i) => xmax*i/4);
  for (const value of ticks) ctx.fillText(attemptAxis ? String(value) : number(value,1), x(value)-10, bottom+19);
  if (attemptAxis) {
    ctx.fillStyle = "#fff0ed"; ctx.fillRect(left, bottom+28, right-left, 20);
    ctx.fillStyle = "#9c291b"; ctx.fillText("Unscored", 4, bottom+42);
  }
  ctx.fillStyle = "#546273";
  ctx.fillText(`${yLabel} (log scale)`, left, 12); ctx.fillText(xLabel, Math.max(left, right - 160), height - 3);
  for (const serie of series) {
    ctx.strokeStyle = serie.color; ctx.fillStyle = serie.color; ctx.lineWidth = 2;
    ctx.beginPath(); let previous = null;
    for (const [px, py] of serie.points) {
      if (!Number.isFinite(px) || !Number.isFinite(py) || py <= 0) { previous = null; continue; }
      if (previous) {
        if (serie.step) ctx.lineTo(x(px), y(previous[1]));
        ctx.lineTo(x(px), y(py));
      } else ctx.moveTo(x(px), y(py));
      previous = [px, py];
    }
    if (serie.line !== false) ctx.stroke();
    if (serie.markers !== false) for (const [px, py] of serie.points.filter(p => p.every(Number.isFinite) && p[1] > 0)) { ctx.beginPath(); ctx.arc(x(px), y(py), 3, 0, Math.PI * 2); ctx.fill(); }
  }
  ctx.strokeStyle = "#b63829";
  for (const value of failures) { const px = x(value), py = bottom + 38; ctx.beginPath(); ctx.moveTo(px-4, py-4); ctx.lineTo(px+4, py+4); ctx.moveTo(px-4, py+4); ctx.lineTo(px+4, py-4); ctx.stroke(); }
  if (!points.length) { ctx.fillStyle = "#546273"; ctx.fillText("No positive values to plot", left + 15, top + 25); }
  chartHover(canvas, {left, top, right, xmax, ...hover});
}

function renderLive() {
  if (!state) return;
  const now = state.server_time + (Date.now() - fetchedAt) / 1000;
  if (isThroughput()) {
    const info = state.segments.find(s => s.segment === $("segment").value);
    const age = info.last_update ? now - info.last_update : null;
    text("connection", !connected ? "Disconnected · showing last received snapshot" : `${info.status} · local snapshot${age !== null ? ` · ${number(age, 0)}s since journal update` : ""}${info.active_trial && age > 300 ? " · stale progress, no recent journal update" : ""}`);
    $("connection").className = !connected || info.status.includes("incomplete") ? "bad" : "good";
    text("live", `Status: ${info.status}. ${info.active_trial ? `Current trial ${info.active_trial}; job phase not recorded. ` : ""}No campaign control is available here.`);
    return;
  }
  const active = state.trials.findLast(t => !t.archived && t.status === "running");
  const stale = active && now - active.time > 15;
  text("connection", !connected ? "Disconnected · showing last received logs" : stale ? "Stale progress · no trial update for over 15 seconds" : "Connected · local logs");
  $("connection").className = !connected || stale ? "bad" : "good";
  const updated = state.last_update ? new Date(state.last_update * 1000).toLocaleString() : "no events";
  const info = state.segments.find(s => s.segment === $("segment").value);
  const archive = info?.archived ? `Viewing archived segment ${info.segment_name}. Recorded search ${info.stopped ? "stopped" : "has no stop event"}. Last archive update: ${info.last_update ? new Date(info.last_update * 1000).toLocaleString() : "no events"}. Current campaign: ` : "";
  text("live", `${archive}${state.stopped ? "Search stopped. " : ""}${active ? `${active.trial} [${active.segment}]: ${active.phase || "setup"}, ${number(now-active.started, 0)}s elapsed. ` : "No active trial. "}Last update: ${updated}`);
}

function render() {
  if (!state) return;
  const segment = $("segment").value, rows = trials(), info = state.segments.find(s => s.segment === segment);
  $("accuracy-view").hidden = isThroughput(); $("throughput-view").hidden = !isThroughput();
  if (isThroughput()) { renderThroughput(info, rows); renderLive(); renderDetails(); return; }
  text("segment-info", info ? `${info.segment_name || info.segment} · ${info.archived ? "Archived" : "Current campaign"} · ${info.profile} · ${info.hardware?.gpu || info.hardware?.cpu || info.hardware?.device || "unknown device"} · Torch ${info.hardware?.torch || "unrecorded"} · ${info.protocol?.train_seconds ?? "?"}s training / ${info.protocol?.total_seconds ?? "?"}s total` : "No segments recorded");
  const best = state.best[segment];
  text("best", best ? `Best retained: ${number(best.score)} m · ${best.trial}` : "Best retained: none");
  text("baseline", `Baselines: ${rows.filter(t => t.baseline && ["kept", "discarded"].includes(t.status)).map(t => `${t.baseline}${t.diagnostic || t.baseline === "small" ? " (diagnostic)" : ""} ${number(t.score)} m`).join("; ") || "none scored"}`);
  const scored = [], retained = [], baselines = [], nonrecords = [], failed = [], attempts = []; let minimum = null, minimumTrial = null;
  rows.forEach((t, index) => {
    const diagnostic = t.diagnostic || t.baseline === "small";
    const record = t.status === "kept" && !diagnostic && Number.isFinite(t.score) && (minimum === null || t.score < minimum);
    if (["kept", "discarded"].includes(t.status)) {
      (record ? t.baseline ? baselines : scored : nonrecords).push([index+1, t.score]);
      if (record) { minimum = t.score; minimumTrial = t.trial; }
    } else if (t.status !== "running") failed.push(index+1);
    if (minimum !== null) retained.push([index+1, minimum]);
    attempts.push([index+1, {
      lines:[`Attempt ${index+1} · ${t.trial}`, `Status: ${t.status}${record ? " · new retained record" : ""}`,
        ...(t.baseline ? [`Baseline: ${t.baseline}${diagnostic ? " (diagnostic)" : ""}`] : diagnostic ? ["Diagnostic trial"] : []),
        `XYZ error: ${Number.isFinite(t.score) ? `${t.score} m${t.score <= 0 ? " (not plotted on log axis)" : ""}` : "unscored"}`,
        `Best retained: ${minimum === null ? "none" : `${minimum} m · ${minimumTrial}`}`, ...(t.reason ? [t.reason] : []), ...(t.phase ? [`Phase: ${t.phase}`] : [])]
    }]);
  });
  chart("history", [{points:scored,color:"#2868b5",line:false},{points:retained,color:"#267044",step:true,markers:false},{points:baselines,color:"#c97c15",line:false},{points:nonrecords,color:"#8793a1",line:false}], "Attempt in segment", "XYZ error (m)", failed, true,
    {context:segment, xmax:rows.length, at:value => nearest(attempts, value)?.[1]});
  $("trials").replaceChildren();
  for (const trial of rows) {
    const row = document.createElement("tr"); if (trial.trial === selected) row.className = "selected";
    const first = document.createElement("td"), button = document.createElement("button"); button.textContent = trial.trial;
    button.addEventListener("click", () => { selected = trial.trial; details = null; detailError = ""; render(); fetchDetails(); }); first.append(button); row.append(first);
    for (const value of [trial.hypothesis || "", trial.status, number(trial.score), `${number(trial.train_seconds,1)} / ${number(trial.total_seconds,1)}`, trial.steps ?? "—"]) { const cell=document.createElement("td"); cell.textContent=value; row.append(cell); }
    $("trials").append(row);
  }
  renderLive(); renderDetails();
}

function renderDetails() {
  if (isThroughput()) { renderThroughputDetails(); return; }
  text("selected", selected ? `Trial ${selected}` : "Trial details");
  const segment = $("segment").value;
  if (details && (details.trial.trial !== selected || details.trial.segment !== segment)) details = null;
  const trial = details?.trial;
  text("detail-summary", trial ? `${trial.status}${trial.reason ? ": " + trial.reason : ""} · ${trial.phase || "complete"}` : detailError || "Select a trial to inspect its training curve and metrics.");
  text("details", details ? JSON.stringify(details, (key, value) => key === "curve" ? undefined : value, 2) : "");
  const best = state?.best[segment];
  const points = curvePoints(details), series = [{points, color:"#2868b5", data:details, label:`Selected trial ${selected}${best?.trial === selected ? " (best retained)" : ""}`}];
  let message = "", legend = selected ? `Blue: selected trial ${selected}. ` : "";
  if ($("compare-best").checked) {
    if (!selected) message = "Select a trial to compare.";
    else if (!best) message = "Comparison unavailable: this segment has no retained best trial.";
    else if (best.trial === selected) message = "The selected trial is the current best retained trial.";
    else if (comparison?.context !== detailContext()) message = "Loading best retained training curve…";
    else if (detailError || comparisonError) message = `Comparison unavailable: ${detailError || comparisonError}`;
    else if (!details) message = "Loading selected trial training curve…";
    else {
      const bestDetails = comparison.data;
      const loss = trial.loss_label?.trim(), bestLoss = bestDetails.trial.loss_label?.trim();
      if (!loss || !bestLoss) message = "Comparison unavailable: loss definition and units were not recorded for both trials.";
      else if (loss !== bestLoss) message = "Comparison unavailable: the trials record different loss definitions or units.";
      else if (!points.some(p => p[1] > 0) || !curvePoints(bestDetails).some(p => p[1] > 0)) message = "Comparison unavailable: one or both trials have no positive training-loss curve values.";
      else {
        series.push({points:curvePoints(bestDetails), color:"#267044", data:bestDetails, label:`Best retained trial ${best.trial}`});
        legend += `Green: best retained trial ${best.trial}. `;
        message = bestDetails.curve_warning || details.curve_warning || "";
      }
    }
  }
  const samples = series.map(s => ({...s, samples:(s.data?.curve || []).filter(p => Number.isFinite(p.train_seconds) && Number.isFinite(p.loss) && p.loss > 0).map(p => [p.train_seconds, p])}));
  chart("curve", series, "Training seconds", "Training objective", [], false, {context:JSON.stringify([detailContext(), series.map(s => s.label)]), at:value => {
    const hits = samples.map(s => ({...s, sample:nearest(s.samples, value)?.[1]})).filter(s => s.sample);
    return hits.length ? {lines:["Nearest recorded training samples", ...hits.flatMap(s => [s.label, `Time: ${s.sample.train_seconds} s · loss: ${s.sample.loss}${Number.isFinite(s.sample.step) ? ` · step: ${s.sample.step}` : ""}`])]} : null;
  }});
  text("curve-legend", `${legend}The y-axis uses a log scale and plots positive values only.`);
  text("comparison-status", message);
  text("loss-label", trial?.loss_label || "Loss definition not recorded. Final validation XYZ determines the comparison.");
}


const shortHash = value => typeof value === "string" ? `${value.slice(0,12)}…` : "not recorded";
const bytes = value => {
  if (!Number.isFinite(value)) return "not recorded";
  const [divisor, unit] = value >= 1024**3 ? [1024**3,"GiB"] : value >= 1024**2 ? [1024**2,"MiB"] : value >= 1024 ? [1024,"KiB"] : [1,"B"];
  return `${number(value / divisor,1)} ${unit}`;
};
const seconds = value => Number.isFinite(value) ? `${number(value)} s` : "not recorded";
const percentage = value => Number.isFinite(value) ? `${number(value * 100,1)}%` : "not recorded";
function fact(label, value) {
  const name = document.createElement("dt"), content = document.createElement("dd"); name.textContent = label; content.textContent = value; $("throughput-facts").append(name, content);
}
function renderThroughput(info, rows) {
  text("segment-info", `${info.segment_name} · ${info.archived ? "Archived" : "Current campaign"} · main training data`);
  text("throughput-objective", info.objective || "Waiting for the frozen throughput campaign");
  text("best", info.best ? `Incumbent trial: ${info.best.trial_id || "qualification baseline"} · source ${shortHash(info.best.source_sha256)}` : "Incumbent unavailable");
  text("baseline", "Qualification timings are baseline calibration, not candidate attempts.");
  const data = info.data || {}, guards = info.guards || {};
  $("throughput-facts").replaceChildren();
  fact("Hardware", [data.gpu || "GPU not recorded", data.threads ? `${data.threads} CPUs` : "CPU allocation not recorded", `RAM ${bytes(data.memory_bytes)}`].join(" · "));
  fact("Workload size", [data.steps ? `${data.steps} updates` : "updates not recorded", data.steps && data.batch_size ? `${data.steps * data.batch_size} windows` : "windows not recorded", data.batch_size ? `batch ${data.batch_size}` : "batch not recorded", data.max_frames ? `${data.max_frames} frames` : "", data.paired_rounds ? `${data.paired_rounds} paired rounds` : ""].filter(Boolean).join(" · "));
  fact("Workloads", info.workloads?.join(" · ") || "not recorded");
  fact("Data / software", `Main training pin ${shortHash(data.revision)} · Torch ${data.torch || "not recorded"} · CUDA ${data.cuda || "not recorded"}`);
  fact("Required paired gains", `Stream noise ${percentage(guards.noise_threshold_fraction)} · sustained noise ${percentage(guards.sustained_noise_threshold_fraction)}. Both aggregate and each paired round must clear their threshold.`);
  fact("Workload guards", ["total", "sustained"].map(metric => `${metric}: ${Object.entries(guards[`per_workload_${metric}_noise_fraction`] || {}).map(([name,value]) => `${name} ≤ ${percentage(value)} regression`).join("; ") || "not recorded"}`).join(". "));
  text("throughput-identity", JSON.stringify({data:info.data, workloads:info.workloads, incumbent:info.best, oracle_sha256:info.oracle_sha256, protocol_sha256:info.protocol_sha256, guards:info.guards}, null, 2));
  const shown = rows.find(t => t.trial === selected) || rows.at(-1);
  const mean = values => { const finite = values.filter(Number.isFinite); return finite.length ? finite.reduce((a,b) => a+b,0) / finite.length : undefined; };
  const rate = mean((shown?.diagnostics || []).map(p => p.candidate?.windows_per_second));
  text("throughput-latest", shown?.measurements.length ? `Trial ${shown.trial} candidate pair means: elapsed ${seconds(mean(shown.measurements.map(p => p.candidate_elapsed_s)))} · stream ${seconds(mean(shown.measurements.map(p => p.candidate_stream_s)))} · sustained ${seconds(mean(shown.measurements.map(p => p.candidate_sustained_s)))} · throughput ${Number.isFinite(rate) ? `${number(rate,1)} windows/s` : "not recorded"}` : shown ? `Trial ${shown.trial}: no sealed timing measurements yet.` : "No candidate measurements yet.");
  text("throughput-counts", `${rows.filter(t => t.measurements.length || t.status === "failed").length} completed or failed attempts · ${rows.filter(t => t.status === "kept").length} kept · ${rows.filter(t => t.decision === "discard").length} discarded · ${rows.filter(t => t.status === "failed").length} failed · ${rows.length} reserved attempts`);
  text("qualification", `Qualification baseline elapsed_s pairs (s): ${JSON.stringify(info.qualification || {})}`);
  const series = [["elapsed_s", "#2868b5"], ["stream_s", "#c97c15"], ["sustained_s", "#267044"]].map(([metric, color]) => ({color, points:rows.map((t,i) => [i+1, t.ratios[metric]?.geometric_ratio]).filter(p => Number.isFinite(p[1]))}));
  series.push({color:"#8793a1", markers:false, points:[[0,1],[Math.max(1,rows.length),1]]});
  const attempts = rows.map((t,i) => [i+1, {lines:[`Attempt ${i+1} · ${t.trial} · ${t.status}`, ...Object.entries(t.ratios).map(([k,v]) => `${k} candidate/incumbent ratio: ${v.geometric_ratio}`)]}]);
  chart("timing-history", series, "Candidate attempt", "Paired seconds ratio", [], true, {context:info.segment, xmax:rows.length, at:value => nearest(attempts,value)?.[1]});
  $("throughput-trials").replaceChildren();
  for (const trial of rows) {
    const row = document.createElement("tr"), first = document.createElement("td"), button = document.createElement("button");
    button.textContent = trial.trial; button.addEventListener("click", () => { selected = trial.trial; details = null; detailError = ""; render(); fetchDetails(); }); first.append(button); row.append(first);
    if (trial.trial === selected) row.className = "selected";
    for (const value of [trial.hypothesis, `${trial.status}${trial.decision ? ` / ${trial.decision}` : " / awaiting decision"}`, ...["elapsed_s", "stream_s", "sustained_s"].map(k => number(trial.ratios[k]?.geometric_ratio))]) { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); }
    $("throughput-trials").append(row);
  }
}
function renderThroughputDetails() {
  if (details && (details.trial.trial !== selected || details.trial.segment !== $("segment").value)) details = null;
  text("throughput-selected", selected ? `Trial ${selected}: same-job workload comparison` : "Workload comparison");
  const trial = details?.trial;
  text("throughput-detail-status", trial ? `${trial.status} · candidate source ${shortHash(trial.source_sha256)} · paired incumbent ${trial.incumbent_trial || "qualification baseline"} source ${shortHash(trial.incumbent_sha256)} · independent gates ${trial.gates ? trial.gates.passed && trial.gates.independent ? "passed" : "not passed" : "not recorded"} · retention metric ${trial.score?.metric || "not recorded"}, eligible ${trial.score?.eligible ?? "unknown"}` : detailError || "Waiting for recorded trial details");
  text("throughput-accounting", trial ? `Job compute ${seconds(trial.job_accounting?.compute_s)} · queue ${seconds(trial.job_accounting?.queue_s)} · reserved cap ${seconds(trial.job_budget_s)}` : "");
  renderDiagnosticSummary(details?.diagnostics || []);
  $("workloads").replaceChildren();
  for (const pair of details?.measurements || []) for (const metric of ["elapsed_s", "stream_s", "sustained_s"]) {
    const candidate = pair[`candidate_${metric}`], incumbent = pair[`incumbent_${metric}`];
    if (candidate === undefined && incumbent === undefined) continue;
    const row = document.createElement("tr");
    for (const value of [`${pair.workload} / ${pair.round+1}`, metric, number(candidate), number(incumbent), number(candidate / incumbent)]) { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); }
    $("workloads").append(row);
  }
  text("throughput-diagnostics", details ? JSON.stringify({source_sha256:trial.source_sha256, incumbent_sha256:trial.incumbent_sha256, reserved_at:trial.reserved_at, recorded_at:trial.recorded_at, score:trial.score, job_accounting:trial.job_accounting, diagnostics:details.diagnostics},null,2) : "No recorded diagnostics yet.");
}

function renderDiagnosticSummary(reports) {
  $("diagnostic-summary").replaceChildren();
  const groups = new Map();
  for (const pair of reports) for (const side of ["candidate", "incumbent"]) {
    const key = `${pair.workload} / ${side}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(pair[side] || {});
  }
  text("diagnostic-summary-note", reports.length ? "Means of available paired-report values. Resource peaks and loader p95 show the maximum reported across pairs." : "Optional timing, loader-stall and resource diagnostics are not recorded yet.");
  for (const [label, records] of groups) {
    const aggregate = (key, peak = false) => {
      const values = records.map(r => r[key]).filter(Number.isFinite);
      return values.length ? peak ? Math.max(...values) : values.reduce((a,b) => a+b,0) / values.length : undefined;
    };
    const values = [label,
      [["Elapsed", "elapsed_s"], ["Setup", "setup_s"], ["Stream", "stream_elapsed_s"], ["Sustained", "sustained_elapsed_s"]].map(([label,key]) => `${label}: ${seconds(aggregate(key))}`).join("\n"),
      `Throughput: ${Number.isFinite(aggregate("windows_per_second")) ? `${number(aggregate("windows_per_second"),1)} windows/s` : "not recorded"}\nStall mean: ${seconds(aggregate("stall_mean_s"))}\nMax pair stall p95: ${seconds(aggregate("stall_p95_s", true))}\nQueue delay mean: ${seconds(aggregate("queue_delay_mean_s"))}`,
      [["RAM", "ram_peak_bytes"], ["GPU", "gpu_peak_bytes"], ["GPU reserved", "gpu_reserved_peak_bytes"], ["Cache", "cache_peak_bytes"], ["Fetched", "bytes_fetched"]].map(([name,key]) => `${name}: ${bytes(aggregate(key,true))}`).join("\n")];
    const row = document.createElement("tr");
    for (const value of values) { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); }
    $("diagnostic-summary").append(row);
  }
}

async function get(url) { const response = await fetch(url, {cache:"no-store"}); const data = await response.json(); if (!response.ok) throw Error(data.error || response.statusText); return data; }
async function fetchDetails() {
  const context = detailContext();
  if (!selected) return;
  if (pendingContext === context) return;
  const request = ++detailRequest;
  pendingContext = context;
  const id = selected, segment = $("segment").value, bestId = state?.best[segment]?.trial;
  const compare = !isThroughput() && $("compare-best").checked && bestId && bestId !== id;
  const ids = compare ? [id, bestId] : [id];
  const results = await Promise.allSettled(ids.map(trialId => get(`/api/trial?segment=${encodeURIComponent(segment)}&id=${encodeURIComponent(trialId)}`).then(data => {
    if (data.trial?.trial !== trialId || data.trial?.segment !== segment) throw Error("trial data does not match the selected segment");
    return data;
  })));
  if (request === detailRequest) pendingContext = null;
  if (request !== detailRequest || context !== detailContext()) return;
  details = results[0].status === "fulfilled" ? results[0].value : null;
  detailError = results[0].status === "rejected" ? `Trial details: ${results[0].reason.message}` : "";
  comparison = compare ? {context, data:results[1].status === "fulfilled" ? results[1].value : null} : null;
  comparisonError = compare && results[1].status === "rejected" ? `best retained trial ${bestId}: ${results[1].reason.message}` : "";
  renderDetails();
}
async function poll() {
  try {
    const next = await get("/api/snapshot"); state = next; fetchedAt = Date.now(); connected = true;
    const previous = $("segment").value; $("segment").replaceChildren();
    for (const s of state.segments) { const option=document.createElement("option"); option.value=s.segment; option.textContent=`${s.segment_name || s.segment} (${s.profile}${s.archived ? ", archived" : ""})`; $("segment").append(option); }
    if (state.segments.some(s => s.segment === previous)) $("segment").value = previous;
    else if (state.segments.length) {
      const requested = new URLSearchParams(location.search).get("segment");
      $("segment").value = state.segments.find(s => s.segment === requested)?.segment || state.segments.at(-1).segment;
    }
    if (!trials().some(t => t.trial === selected)) selected = trials().at(-1)?.trial || null;
    text("warning", state.warnings.join(" · ")); render(); fetchDetails();
  } catch (error) { connected = false; text("warning", error.message); if (!state) text("connection", "Disconnected · unable to read logs"); renderLive(); }
  setTimeout(poll, 2000);
}
$("segment").addEventListener("change", () => { const url = new URL(location.href); url.searchParams.set("segment", $("segment").value); history.replaceState(null, "", url); selected = trials().at(-1)?.trial || null; details = null; detailError = ""; render(); fetchDetails(); });
$("compare-best").addEventListener("change", () => { renderDetails(); fetchDetails(); });
window.addEventListener("resize", () => { clearChartHovers(); render(); }); window.addEventListener("scroll", clearChartHovers, true); setInterval(renderLive, 1000); poll();
