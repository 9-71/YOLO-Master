import { useEffect, useMemo, useState } from "react";
import { absUrl } from "../api/client";
import { useJobs } from "../state/JobContext";

export function ArtifactInspector() {
  const { baseUrl, selectedJobId, artifacts, refreshArtifacts } = useJobs();
  const [lightbox, setLightbox] = useState<{ url: string; filename: string } | null>(null);
  useEffect(() => { const close = (event: KeyboardEvent) => { if (event.key === "Escape") setLightbox(null); }; document.addEventListener("keydown", close); return () => document.removeEventListener("keydown", close); }, []);
  const images = useMemo(() => {
    const list = artifacts.artifacts.filter((entry) => entry.is_image).map((entry) => ({ filename: entry.filename, url: absUrl(baseUrl, entry.download_url) }));
    for (const artifactId of artifacts.image_artifacts) { const filename = artifactId.split("/").pop(); const encodedId = artifactId.split("/").map(encodeURIComponent).join("/"); if (filename && selectedJobId) list.push({ filename, url: absUrl(baseUrl, `/static/artifacts/${encodeURIComponent(selectedJobId)}/${encodedId}`) }); }
    return [...new Map(list.map((item) => [item.url, item])).values()];
  }, [artifacts, baseUrl, selectedJobId]);
  const files = artifacts.artifacts.filter((entry) => !entry.is_image);
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900">
      <div className="flex items-center gap-3 border-b border-slate-800 px-4 py-3"><h2 className="text-sm font-bold uppercase tracking-wider text-slate-300">Artifact Inspector</h2><span className="text-xs text-slate-500">{images.length + files.length ? `${images.length + files.length} file(s)` : ""}</span><button type="button" disabled={!selectedJobId} onClick={() => void refreshArtifacts(true)} className="ml-auto rounded-md border border-slate-700 px-2.5 py-1 text-xs font-semibold hover:bg-slate-800 disabled:opacity-40">Refresh</button></div>
      <div className="max-h-96 overflow-y-auto p-3"><div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-4">{images.map((image) => <figure key={image.url} className="group overflow-hidden rounded-md border border-slate-800 bg-slate-950"><img src={image.url} alt={image.filename} loading="lazy" onClick={() => setLightbox(image)} className="h-28 w-full cursor-zoom-in object-cover group-hover:opacity-80" /><figcaption className="flex items-center gap-2 border-t border-slate-800 px-2 py-1"><span title={image.filename} className="flex-1 truncate font-mono text-[10px] text-slate-400">{image.filename}</span><a href={image.url} target="_blank" rel="noreferrer" className="text-[10px] font-semibold text-cyan-400 hover:underline">Open</a></figcaption></figure>)}</div>
        <ul className="mt-2 space-y-1">{files.map((entry) => { const url = absUrl(baseUrl, entry.download_url); return <li key={entry.artifact_id} className="flex items-center gap-2 rounded-md border border-slate-800 bg-slate-950 px-3 py-2"><span title={entry.artifact_id} className="flex-1 truncate font-mono text-xs text-slate-300">{entry.artifact_id}</span><a href={url} target="_blank" rel="noreferrer" className="text-xs font-semibold text-cyan-400 hover:underline">View</a><a href={url} download={entry.filename} className="text-xs font-semibold text-cyan-400 hover:underline">Download</a></li>; })}</ul>
        {images.length + files.length === 0 && <p className="py-6 text-center text-xs text-slate-500">No artifacts yet — they appear once the selected job completes.</p>}
      </div>
      {lightbox && <div onClick={() => setLightbox(null)} className="fixed inset-0 z-50 flex cursor-zoom-out items-center justify-center bg-black/90 p-6" role="dialog" aria-modal="true" aria-label={`Preview ${lightbox.filename}`}><img src={lightbox.url} alt={lightbox.filename} className="max-h-[90vh] max-w-full rounded-md border border-slate-700 object-contain" /></div>}
    </section>
  );
}
