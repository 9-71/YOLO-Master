import { useState, type FormEvent } from "react";
import { useJobs } from "../state/JobContext";
import type { JobRequest, TaskType } from "../types";

const TASK_PRESETS: Record<TaskType, { modelPath: string; dataSource: string; outputDir: string }> = {
  predict: { modelPath: "./ckpts/yolov8n.pt", dataSource: "ultralytics/assets/bus.jpg", outputDir: "runs/predict" },
  train: { modelPath: "./ckpts/yolov8n.pt", dataSource: "coco8.yaml", outputDir: "runs/train" },
  val: { modelPath: "./ckpts/yolov8n.pt", dataSource: "coco8.yaml", outputDir: "runs/val" },
  export: { modelPath: "./ckpts/yolov8n.pt", dataSource: "", outputDir: "runs/export" },
  diagnose: { modelPath: "", dataSource: "", outputDir: "runs/diagnose" },
};

function buildJobId(taskType: TaskType): string {
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "").replace("T", "_");
  const suffix = crypto.randomUUID().replace(/-/g, "").slice(0, 8);
  return `${taskType}_${stamp}_${suffix}`;
}

const fieldClass = "w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-xs focus:border-cyan-500 focus:outline-none";

export function DispatchForm() {
  const { submitJob } = useJobs();
  const [taskType, setTaskType] = useState<TaskType>("predict");
  const [modelPath, setModelPath] = useState(TASK_PRESETS.predict.modelPath);
  const [dataSource, setDataSource] = useState(TASK_PRESETS.predict.dataSource);
  const [outputDir, setOutputDir] = useState(TASK_PRESETS.predict.outputDir);
  const [device, setDevice] = useState("0"); const [conf, setConf] = useState("0.25");
  const [allowedPaths, setAllowedPaths] = useState("., ultralytics/assets, runs, ckpts");
  const [extraParams, setExtraParams] = useState(""); const [feedback, setFeedback] = useState(""); const [busy, setBusy] = useState(false);
  const needsData = ["predict", "train", "val"].includes(taskType);
  const needsDevice = needsData;

  function changeTask(value: TaskType) {
    setTaskType(value); const preset = TASK_PRESETS[value]; setModelPath(preset.modelPath); setDataSource(preset.dataSource); setOutputDir(preset.outputDir); setFeedback("");
  }

  async function submit(event: FormEvent) {
    event.preventDefault(); setFeedback(""); let extras: Record<string, unknown> = {};
    try {
      if (extraParams.trim()) {
        const parsed: unknown = JSON.parse(extraParams);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("top-level value must be a JSON object");
        extras = parsed as Record<string, unknown>;
      }
      const params = { ...extras };
      if (needsData) Object.assign(params, { model_path: modelPath.trim(), data_source: dataSource.trim(), conf: Number.parseFloat(conf), device: device.trim() || "cpu" });
      else if (taskType === "export") Object.assign(params, { model_path: modelPath.trim(), format: params.format || "onnx" });
      const payload: JobRequest = {
        job_id: buildJobId(taskType), task_type: taskType, params, output: { output_dir: outputDir.trim() },
        security_constraints: { allowed_paths: allowedPaths.split(",").map((item) => item.trim()).filter(Boolean) },
        runtime_tracking: { timeout_seconds: 300, cancellable: true },
      };
      setBusy(true); await submitJob(payload); setFeedback(`✓ Job ${payload.job_id} submitted (201)`);
    } catch (error) { setFeedback(`✗ ${error instanceof Error ? error.message : String(error)}`); }
    finally { setBusy(false); }
  }

  return (
    <aside>
      <section className="rounded-lg border border-slate-800 bg-slate-900">
        <div className="border-b border-slate-800 px-4 py-3"><h2 className="text-sm font-bold uppercase tracking-wider text-slate-300">Job Dispatch</h2><p className="mt-0.5 text-xs text-slate-500">POST /api/v1/jobs — payload follows core/schema.py</p></div>
        <form className="space-y-4 px-4 py-4" onSubmit={submit}>
          <label className="block text-xs font-semibold text-slate-400">Task Type<select value={taskType} onChange={(event) => changeTask(event.target.value as TaskType)} className={`${fieldClass} mt-1 text-sm font-sans`}><option>predict</option><option>val</option><option>diagnose</option><option>train</option><option>export</option></select></label>
          {taskType !== "diagnose" && <label className="block text-xs font-semibold text-slate-400">Model Path<input value={modelPath} onChange={(event) => setModelPath(event.target.value)} className={`${fieldClass} mt-1`} /></label>}
          {needsData && <label className="block text-xs font-semibold text-slate-400">Data Source<input value={dataSource} onChange={(event) => setDataSource(event.target.value)} placeholder="image / video / directory / dataset YAML" className={`${fieldClass} mt-1`} /></label>}
          {needsDevice && <label className="block text-xs font-semibold text-slate-400">Device<input value={device} onChange={(event) => setDevice(event.target.value)} className={`${fieldClass} mt-1`} /><span className="mt-1 block text-[11px] font-normal text-slate-500">“0” (CUDA), “cpu”, “mps”, or “0,1”</span></label>}
          <details className="rounded-md border border-slate-800 bg-slate-900/60 px-3 py-2">
            <summary className="cursor-pointer select-none text-xs font-semibold text-slate-400 hover:text-slate-300">Advanced options</summary>
            <div className="mt-3 space-y-4">
              {needsData && <label className="block text-xs font-semibold text-slate-400">Confidence Threshold<input type="number" min="0.01" max="1" step="0.01" value={conf} onChange={(event) => setConf(event.target.value)} className={`${fieldClass} mt-1`} /></label>}
              <label className="block text-xs font-semibold text-slate-400">Output Directory<input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} className={`${fieldClass} mt-1`} /></label>
              <label className="block text-xs font-semibold text-slate-400">Allowed Paths <span className="font-normal text-slate-500">(comma-separated)</span><input value={allowedPaths} onChange={(event) => setAllowedPaths(event.target.value)} className={`${fieldClass} mt-1`} /></label>
              <label className="block text-xs font-semibold text-slate-400">Extra Params <span className="font-normal text-slate-500">(JSON object)</span><textarea rows={3} value={extraParams} onChange={(event) => setExtraParams(event.target.value)} placeholder={'e.g. {"epochs": 5, "imgsz": 320}'} className={`${fieldClass} mt-1`} /></label>
            </div>
          </details>
          <button disabled={busy} className="w-full rounded-md bg-cyan-600 px-4 py-2.5 text-sm font-bold text-white hover:bg-cyan-500 disabled:opacity-50">{busy ? "Dispatching…" : "Dispatch Job"}</button>
          {feedback && <p className={`text-xs font-semibold ${feedback.startsWith("✓") ? "text-emerald-400" : "text-rose-400"}`}>{feedback}</p>}
        </form>
      </section>
    </aside>
  );
}
