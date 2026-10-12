export async function getHealth() {
  const response = await fetch("/api/health");
  if (!response.ok) throw new Error("Health check failed");
  return response.json();
}

export async function getResults() {
  const response = await fetch("/api/results");
  if (!response.ok) throw new Error("Could not load saved results");
  return response.json();
}

export async function analyzeQuestion(question) {
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).join(" ")
      : detail || "Analysis failed";
    throw new Error(message);
  }
  return body;
}

export function figureUrl(name) {
  return `/api/figures/${name}`;
}
