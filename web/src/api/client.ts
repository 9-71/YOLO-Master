const HTTP_FRIENDLY: Record<number, string> = {
  400: "Bad request",
  404: "Not found",
  409: "Conflict",
  422: "Validation failed",
};

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

function extractDetail(body: unknown): string {
  if (!body || typeof body !== "object" || !("detail" in body)) return "No error details returned";
  const detail = (body as { detail: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      if (!item || typeof item !== "object") return String(item);
      const entry = item as { loc?: unknown[]; msg?: string };
      const loc = (entry.loc ?? []).filter((part) => part !== "body").join(".");
      return `${loc ? `${loc}: ` : ""}${entry.msg ?? "invalid value"}`;
    }).join("; ");
  }
  return JSON.stringify(detail);
}

export async function api<T>(baseUrl: string, path: string, options: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl}${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...options.headers },
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") throw error;
    const reason = error instanceof Error ? error.message : String(error);
    throw new ApiError(0, `Cannot reach engine at ${baseUrl} — ${reason}`);
  }
  if (!response.ok) {
    let body: unknown = null;
    try { body = await response.json(); } catch { /* response may not be JSON */ }
    const label = HTTP_FRIENDLY[response.status] ?? "Request failed";
    throw new ApiError(response.status, `${label} (HTTP ${response.status}): ${extractDetail(body)}`);
  }
  return response.json() as Promise<T>;
}

export function absUrl(baseUrl: string, path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  return `${baseUrl}${path.startsWith("/") ? path : `/${path}`}`;
}
