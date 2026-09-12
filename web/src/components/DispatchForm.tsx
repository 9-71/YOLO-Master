import { useState, type FormEvent } from "react";
import { useJobs } from "../state/JobContext";
import type { JobRequest, TaskType } from "../types";

/**
 * Task-specific New Job form. Every params key maps 1:1 to the corresponding
 * production handler contract in f1/handlers/*.py — there is deliberately no
 * free-form JSON escape hatch:
 *   train:    model_path*, data_source*(yaml), epochs*(int>0),
 *             batch_size?(int>0), seed?(int>=0), device?
 *   predict:  model_path*, data_source*, conf?(0,1], batch_size?(int>0), device?
 *   val:      model_path*, data_source*(yaml), imgsz?(int>0),
 *             batch_size?(int>0), conf?(0,1], device?
 *   export:   model_path*, format(allowlist), imgsz?(int>0), device?,
 *             half?(bool), int8?(bool)
 *   diagnose: params must be {}
 * Optional numeric fields are OMITTED when left blank.
 */

interface TaskPreset {
  model: string;
  data: string;
  output: string;
}

const TASK_PRESETS: Record<TaskType, TaskPreset> = {
  predict: { model: "./ckpts/yolov8n.pt", data: "ultralytics/assets/bus.jpg", output: "runs/predict" },
  train: { model: "./ckpts/yolov8n.pt", data: "coco8.yaml", output: "runs/train" },
  val: { model: "./ckpts/yolov8n.pt", data: "coco8.yaml", output: "runs/val" },
  export: { model: "./ckpts/yolov8n.pt", data: "", output: "runs/export" },
  diagnose: { model: "", data: "", output: "runs/diagnose" },
};

// Mirrors f1/handlers/export.py SUPPORTED_EXPORT_FORMATS (closed allowlist).
const EXPORT_FORMATS = [
  "onnx",
  "torchscript",
  "openvino",
  "engine",
  "coreml",
  "saved_model",
  "pb",
  "tflite",
  "edgetpu",
  "tfjs",
  "paddle",
  "ncnn",
  "imx",
  "rknn",
  "mnn",
] as const;

/** Fixed fail-closed path policy; the UI no longer allows editing it. */
const FIXED_ALLOWED_PATHS = [".", "ultralytics/assets", "runs", "ckpts"];

const TASK_ORDER: TaskType[] = ["predict", "train", "val", "export", "diagnose"];

function buildJobId(taskType: TaskType): string {
  const stamp = new Date()
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(/\.\d+Z$/, "")
    .replace("T", "_");
  const suffix = crypto.randomUUID().replace(/-/g, "").slice(0, 8);
  return `${taskType}_${stamp}_${suffix}`;
}

/** Strict integer parse: digits only (optional leading minus), finite/safe. */
function parseInteger(raw: string, field: string): number {
  const text = raw.trim();
  if (!/^-?\d+$/.test(text)) throw new Error(`${field} must be a whole number, got "${raw}"`);
  const value = Number(text);
  if (!Number.isSafeInteger(value)) throw new Error(`${field} is outside the safe integer range`);
  return value;
}

function requirePositiveInt(raw: string, field: string): number {
  const value = parseInteger(raw, field);
  if (value <= 0) throw new Error(`${field} must be > 0, got ${value}`);
  return value;
}

function parseOptionalPositiveInt(raw: string, field: string): number | undefined {
  if (!raw.trim()) return undefined;
  return requirePositiveInt(raw, field);
}

function parseOptionalNonNegativeInt(raw: string, field: string): number | undefined {
  if (!raw.trim()) return undefined;
  const value = parseInteger(raw, field);
  if (value < 0) throw new Error(`${field} must be >= 0, got ${value}`);
  return value;
}

function parseOptionalConf(raw: string, field: string): number | undefined {
  const text = raw.trim();
  if (!text) return undefined;
  if (!/^-?\d*(?:\.\d+)?$/.test(text)) throw new Error(`${field} must be a number, got "${raw}"`);
  const value = Number(text);
  if (!Number.isFinite(value)) throw new Error(`${field} must be a finite number`);
  if (!(value > 0 && value <= 1)) throw new Error(`${field} must be in range (0.0, 1.0], got ${value}`);
  return value;
}

const fieldClass =
  "w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-xs focus:border-cyan-500 focus:outline-none";
const labelClass = "block text-xs font-semibold text-slate-400";
const hintClass = "mt-1 block text-[11px] font-normal text-slate-500";
const optionalDefaultHint = "Leave blank to use server/engine default.";

export function DispatchForm({ onJobCreated }: { onJobCreated: (jobId: string) => void }) {
  const { submitJob } = useJobs();
  const [taskType, setTaskType] = useState<TaskType>("predict");
  const [modelPath, setModelPath] = useState(TASK_PRESETS.predict.model);
  const [dataSource, setDataSource] = useState(TASK_PRESETS.predict.data);
  const [outputDir, setOutputDir] = useState(TASK_PRESETS.predict.output);
  const [epochs, setEpochs] = useState("10");
  const [batchSize, setBatchSize] = useState("");
  const [seed, setSeed] = useState("");
  const [conf, setConf] = useState("");
  const [imgsz, setImgsz] = useState("");
  const [device, setDevice] = useState("");
  const [format, setFormat] = useState<string>("onnx");
  const [half, setHalf] = useState(false);
  const [int8, setInt8] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [busy, setBusy] = useState(false);

  const preset = TASK_PRESETS[taskType];
  const needsModel = taskType !== "diagnose";
  const needsData = taskType === "predict" || taskType === "train" || taskType === "val";
  const isTrain = taskType === "train";
  const isVal = taskType === "val";
  const isPredict = taskType === "predict";
  const isExport = taskType === "export";
  const isDiagnose = taskType === "diagnose";

  function changeTask(value: TaskType) {
    const next = TASK_PRESETS[value];
    setTaskType(value);
    setModelPath(next.model);
    setDataSource(next.data);
    setOutputDir(next.output);
    setEpochs("10");
    setBatchSize("");
    setSeed("");
    setConf("");
    setImgsz("");
    setDevice("");
    setFormat("onnx");
    setHalf(false);
    setInt8(false);
    setFeedback("");
  }

  function resetField(setter: (value: string) => void, presetValue: string) {
    setter(presetValue);
    setFeedback("");
  }

  function FieldReset({ current, presetValue, onReset }: { current: string; presetValue: string; onReset: () => void }) {
    return (
      <button
        type="button"
        onClick={onReset}
        disabled={current === presetValue}
        title="Reset to task default"
        className="rounded-md border border-slate-700 px-2 py-1 text-[10px] font-semibold text-slate-400 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
      >
        Reset
      </button>
    );
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setFeedback("");
    try {
      const params: Record<string, unknown> = {};
      const model = modelPath.trim();
      const data = dataSource.trim();
      const output = outputDir.trim();

      if (!output) throw new Error("Output directory is required");

      if (needsModel && !model) throw new Error("Model Path is required");
      if (needsData) {
        if (!data) throw new Error("Data Source is required");
        if ((isTrain || isVal) && ![".yaml", ".yml"].some((ext) => data.toLowerCase().endsWith(ext))) {
          throw new Error(`${taskType} Data Source must be a dataset YAML file (e.g., coco8.yaml)`);
        }
      }

      if (isPredict) {
        Object.assign(params, { model_path: model, data_source: data });
        const confValue = parseOptionalConf(conf, "Confidence Threshold");
        const batchValue = parseOptionalPositiveInt(batchSize, "Batch Size");
        if (confValue !== undefined) params.conf = confValue;
        if (batchValue !== undefined) params.batch_size = batchValue;
      } else if (isTrain) {
        Object.assign(params, {
          model_path: model,
          data_source: data,
          epochs: requirePositiveInt(epochs, "Epochs"),
        });
        const batchValue = parseOptionalPositiveInt(batchSize, "Batch Size");
        const seedValue = parseOptionalNonNegativeInt(seed, "Seed");
        if (batchValue !== undefined) params.batch_size = batchValue;
        if (seedValue !== undefined) params.seed = seedValue;
        // imgsz/conf are intentionally never submitted for train.
      } else if (isVal) {
        Object.assign(params, { model_path: model, data_source: data });
        const imgszValue = parseOptionalPositiveInt(imgsz, "Image Size");
        const batchValue = parseOptionalPositiveInt(batchSize, "Batch Size");
        const confValue = parseOptionalConf(conf, "Confidence Threshold");
        if (imgszValue !== undefined) params.imgsz = imgszValue;
        if (batchValue !== undefined) params.batch_size = batchValue;
        if (confValue !== undefined) params.conf = confValue;
      } else if (isExport) {
        params.model_path = model;
        params.format = format;
        const imgszValue = parseOptionalPositiveInt(imgsz, "Image Size");
        if (imgszValue !== undefined) params.imgsz = imgszValue;
        if (half) params.half = true;
        if (int8) params.int8 = true;
      }
      // diagnose: params stays exactly {}.

      const deviceValue = device.trim();
      if (deviceValue && !isDiagnose) params.device = deviceValue;

      const payload: JobRequest = {
        job_id: buildJobId(taskType),
        task_type: taskType,
        params,
        output: { output_dir: output },
        security_constraints: { allowed_paths: FIXED_ALLOWED_PATHS },
        runtime_tracking: { timeout_seconds: 300, cancellable: true },
      };
      setBusy(true);
      const jobId = await submitJob(payload);
      setFeedback(`✓ Job ${jobId} submitted (201) — opening detail…`);
      onJobCreated(jobId);
    } catch (error) {
      setFeedback(`✗ ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl">
      <section className="rounded-lg border border-slate-800 bg-slate-900">
        <div className="border-b border-slate-800 px-4 py-3">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300">New Job</h2>
          <p className="mt-0.5 text-xs text-slate-500">POST /api/v1/jobs — task-specific payload validated against the handler contract</p>
        </div>
        <form className="space-y-4 px-4 py-4" onSubmit={submit} noValidate>
          <label className={labelClass}>
            Task Type
            <select
              value={taskType}
              onChange={(event) => changeTask(event.target.value as TaskType)}
              className={`${fieldClass} mt-1 text-sm font-sans`}
            >
              {TASK_ORDER.map((task) => (
                <option key={task}>{task}</option>
              ))}
            </select>
          </label>

          {needsModel && (
            <div>
              <div className="flex items-center justify-between gap-2">
                <label className={`${labelClass} flex-1`}>
                  Model Path
                  <input value={modelPath} onChange={(event) => setModelPath(event.target.value)} className={`${fieldClass} mt-1`} />
                </label>
                <div className="pt-5">
                  <FieldReset current={modelPath} presetValue={preset.model} onReset={() => resetField(setModelPath, preset.model)} />
                </div>
              </div>
            </div>
          )}

          {needsData && (
            <div className="flex items-center justify-between gap-2">
              <label className={`${labelClass} flex-1`}>
                Data Source
                <input
                  value={dataSource}
                  onChange={(event) => setDataSource(event.target.value)}
                  placeholder={isPredict ? "image / video / directory / dataset YAML" : "dataset YAML, e.g. coco8.yaml"}
                  className={`${fieldClass} mt-1`}
                />
              </label>
              <div className="pt-5">
                <FieldReset current={dataSource} presetValue={preset.data} onReset={() => resetField(setDataSource, preset.data)} />
              </div>
            </div>
          )}

          {isTrain && (
            <label className={labelClass}>
              Epochs <span className="font-normal text-rose-400">*</span>
              <input type="number" min={1} step={1} value={epochs} onChange={(event) => setEpochs(event.target.value)} className={`${fieldClass} mt-1`} />
              <span className={hintClass}>Required, whole number &gt; 0</span>
            </label>
          )}

          {(isPredict || isTrain || isVal) && (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <label className={labelClass}>
                Batch Size <span className="font-normal text-slate-500">(optional)</span>
                <input type="number" min={1} step={1} value={batchSize} onChange={(event) => setBatchSize(event.target.value)} placeholder={optionalDefaultHint} className={`${fieldClass} mt-1`} />
              </label>
              <label className={labelClass}>
                Device <span className="font-normal text-slate-500">(optional)</span>
                <input value={device} onChange={(event) => setDevice(event.target.value)} placeholder="cpu / 0 / mps / 0,1" className={`${fieldClass} mt-1`} />
                <span className={hintClass}>{optionalDefaultHint}</span>
              </label>
            </div>
          )}

          {isTrain && (
            <label className={labelClass}>
              Seed <span className="font-normal text-slate-500">(optional)</span>
              <input type="number" min={0} step={1} value={seed} onChange={(event) => setSeed(event.target.value)} placeholder={optionalDefaultHint} className={`${fieldClass} mt-1`} />
              <span className={hintClass}>Whole number &ge; 0; explicit values are passed to the training engine. {optionalDefaultHint}</span>
            </label>
          )}

          {(isPredict || isVal) && (
            <label className={labelClass}>
              Confidence Threshold <span className="font-normal text-slate-500">(optional)</span>
              <input type="number" min="0.01" max={1} step="0.01" value={conf} onChange={(event) => setConf(event.target.value)} placeholder={optionalDefaultHint} className={`${fieldClass} mt-1`} />
              <span className={hintClass}>Range (0.0, 1.0]; omitted when blank. {optionalDefaultHint}</span>
            </label>
          )}

          {(isVal || isExport) && (
            <label className={labelClass}>
              Image Size <span className="font-normal text-slate-500">(optional)</span>
              <input type="number" min={1} step={1} value={imgsz} onChange={(event) => setImgsz(event.target.value)} placeholder={optionalDefaultHint} className={`${fieldClass} mt-1`} />
            </label>
          )}

          {isExport && (
            <>
              <label className={labelClass}>
                Export Format
                <select value={format} onChange={(event) => setFormat(event.target.value)} className={`${fieldClass} mt-1 font-mono`}>
                  {EXPORT_FORMATS.map((item) => (
                    <option key={item}>{item}</option>
                  ))}
                </select>
                <span className={hintClass}>Closed allowlist enforced by the export handler</span>
              </label>
              <label className={labelClass}>
                Device <span className="font-normal text-slate-500">(optional)</span>
                <input value={device} onChange={(event) => setDevice(event.target.value)} placeholder="cpu / 0 / mps" className={`${fieldClass} mt-1`} />
                <span className={hintClass}>{optionalDefaultHint}</span>
              </label>
              <div className="flex gap-6">
                <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-400">
                  <input type="checkbox" checked={half} onChange={(event) => setHalf(event.target.checked)} className="accent-cyan-500" />
                  half (FP16)
                </label>
                <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-400">
                  <input type="checkbox" checked={int8} onChange={(event) => setInt8(event.target.checked)} className="accent-cyan-500" />
                  int8 quantization
                </label>
              </div>
            </>
          )}

          {isDiagnose && (
            <div className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-[11px] text-slate-500">
              Diagnose reads system state only — <span className="font-mono text-slate-400">params={"{}"}</span> is enforced.
            </div>
          )}

          <label className={labelClass}>
            Output Directory
            <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} className={`${fieldClass} mt-1`} />
            <span className={hintClass}>Submitted as output.output_dir; artifacts are isolated per job id</span>
          </label>

          <div className="rounded-md border border-slate-800 bg-slate-900/60 px-3 py-2 text-[11px] text-slate-500">
            <span className="font-semibold text-slate-400">Fixed path policy:</span>{" "}
            <span className="font-mono text-slate-400">{FIXED_ALLOWED_PATHS.join(", ")}</span>
            <span> — security_constraints.allowed_paths is fail-closed and not editable</span>
          </div>

          <button
            disabled={busy}
            className="w-full rounded-md bg-cyan-600 px-4 py-2.5 text-sm font-bold text-white hover:bg-cyan-500 disabled:opacity-50"
          >
            {busy ? "Dispatching…" : "Dispatch Job"}
          </button>
          {feedback && (
            <p className={`text-xs font-semibold ${feedback.startsWith("✓") ? "text-emerald-400" : "text-rose-400"}`}>
              {feedback}
            </p>
          )}
        </form>
      </section>
    </div>
  );
}
