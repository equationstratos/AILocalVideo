import { useEffect, useRef, useState } from "react";
import { fetchStyles, startGeneration, fetchJob, videoUrl } from "./api.js";

export default function App() {
  const [styles, setStyles] = useState([]);
  const [prompt, setPrompt] = useState("");
  const [style, setStyle] = useState("");
  const [numFrames, setNumFrames] = useState("");
  const [steps, setSteps] = useState("");
  const [seed, setSeed] = useState("");

  const [job, setJob] = useState(null); // {job_id, status, progress, message}
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    fetchStyles()
      .then((s) => {
        setStyles(s);
        if (s.length) setStyle(s[0].name);
      })
      .catch((e) => setError(e.message));
    return () => clearInterval(pollRef.current);
  }, []);

  const selected = styles.find((s) => s.name === style);
  const busy = job && (job.status === "queued" || job.status === "running");

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    setJob(null);
    clearInterval(pollRef.current);
    try {
      const payload = { prompt, style };
      if (numFrames) payload.num_frames = Number(numFrames);
      if (steps) payload.steps = Number(steps);
      if (seed) payload.seed = Number(seed);
      const { job_id } = await startGeneration(payload);
      setJob({ job_id, status: "queued", progress: 0 });
      pollRef.current = setInterval(() => poll(job_id), 1500);
    } catch (e) {
      setError(e.message);
    }
  }

  async function poll(jobId) {
    try {
      const status = await fetchJob(jobId);
      setJob(status);
      if (status.status === "done" || status.status === "error") {
        clearInterval(pollRef.current);
      }
    } catch (e) {
      clearInterval(pollRef.current);
      setError(e.message);
    }
  }

  return (
    <div className="container">
      <h1>🎬 AILocalVideo</h1>
      <p className="subtitle">
        Génération de vidéos par IA, 100 % local. Sur CPU, soyez patient — un
        rendu peut prendre de longues minutes.
      </p>

      <form onSubmit={onSubmit} className="card">
        <label>
          Description (prompt)
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="ex : un chat qui surfe sur une vague au coucher du soleil"
            rows={3}
            required
          />
        </label>

        <label>
          Style
          <select value={style} onChange={(e) => setStyle(e.target.value)}>
            {styles.map((s) => (
              <option key={s.name} value={s.name}>
                {s.label} ({s.backend})
              </option>
            ))}
          </select>
        </label>

        <details className="advanced">
          <summary>Réglages avancés</summary>
          <div className="grid">
            <label>
              Frames
              <input
                type="number"
                min="1"
                value={numFrames}
                onChange={(e) => setNumFrames(e.target.value)}
                placeholder={selected?.defaults.num_frames}
              />
            </label>
            <label>
              Étapes (steps)
              <input
                type="number"
                min="1"
                value={steps}
                onChange={(e) => setSteps(e.target.value)}
                placeholder={selected?.defaults.steps}
              />
            </label>
            <label>
              Seed
              <input
                type="number"
                min="0"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                placeholder="aléatoire"
              />
            </label>
          </div>
        </details>

        <button type="submit" disabled={busy || !prompt}>
          {busy ? "Génération en cours…" : "Générer la vidéo"}
        </button>
      </form>

      {error && <div className="error">⚠️ {error}</div>}

      {job && (
        <div className="card status">
          <p>
            Statut : <strong>{job.status}</strong>
          </p>
          {busy && (
            <div className="progress">
              <div
                className="bar"
                style={{ width: `${Math.round((job.progress || 0) * 100)}%` }}
              />
            </div>
          )}
          {job.status === "error" && (
            <div className="error">Échec : {job.message}</div>
          )}
          {job.status === "done" && job.has_video && (
            <div className="result">
              <video src={videoUrl(job.job_id)} controls loop autoPlay />
              <a href={videoUrl(job.job_id)} download>
                ⬇️ Télécharger
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
