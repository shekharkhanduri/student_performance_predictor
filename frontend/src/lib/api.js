const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
// All routes live under the versioned prefix.
const API_V1 = `${API_BASE}/api/v1`;

async function request(path, options = {}) {
  const response = await fetch(`${API_V1}${path}`, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail =
      typeof payload === "object" && payload?.detail ? payload.detail : payload;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  return payload;
}

export async function getHealth() {
  return request("/health");
}

export async function getStudents(params = {}) {
  const query = new URLSearchParams();
  if (params.limit)        query.set("limit",        String(params.limit));
  if (params.offset)       query.set("offset",       String(params.offset));
  if (params.risk_level)   query.set("risk_level",   params.risk_level);
  if (params.dataset_type) query.set("dataset_type", params.dataset_type);
  const suffix = query.toString() ? `?${query}` : "";
  return request(`/students${suffix}`);
}

export async function getStudent(studentId, datasetType) {
  const query = new URLSearchParams();
  if (datasetType) query.set("dataset_type", datasetType);
  const suffix = query.toString() ? `?${query}` : "";
  return request(`/student/${studentId}${suffix}`);
}

export async function createPrediction(payload) {
  return request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function uploadStudentsCsv(file) {
  const formData = new FormData();
  formData.append("file", file);
  return request("/upload", { method: "POST", body: formData });
}

// Exported for display in the Header component.
export { API_BASE };
