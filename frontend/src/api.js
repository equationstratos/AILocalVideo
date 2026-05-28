// Petit wrapper autour de l'API REST du backend.

async function asJson(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Erreur ${res.status}`);
  }
  return res.json();
}

export function fetchStyles() {
  return fetch("/api/styles").then(asJson);
}

export function startGeneration(payload) {
  return fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then(asJson);
}

export function fetchJob(jobId) {
  return fetch(`/api/jobs/${jobId}`).then(asJson);
}

export function videoUrl(jobId) {
  return `/api/jobs/${jobId}/video`;
}
