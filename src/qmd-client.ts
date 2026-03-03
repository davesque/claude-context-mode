/**
 * QMD HTTP Client — delegates index/search to a running QMD daemon.
 *
 * Context-mode uses this to route session content through QMD's hybrid search
 * pipeline (BM25 + vector + RRF + reranking) instead of its built-in FTS5-only
 * search. Session content lives in a `_ctx/<sessionId>` collection and is
 * flushed when the session ends.
 *
 * QMD must be running as an HTTP daemon (`qmd mcp --http --daemon`).
 * Configure the URL via QMD_URL env var (default: http://localhost:8181).
 */

export interface QmdSearchResult {
  docid: string;
  file: string;
  title: string;
  score: number;
  context: string | null;
  snippet: string;
}

export interface QmdIndexResult {
  embedded: number;
  collection: string;
}

export class QmdClient {
  private baseUrl: string;
  private sessionId: string;
  private collection: string;
  private indexCounter = 0;

  constructor(baseUrl: string, sessionId: string) {
    // Strip trailing slash
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.sessionId = sessionId;
    this.collection = `_ctx/${sessionId}`;
  }

  /**
   * Check if QMD daemon is reachable.
   */
  async isAvailable(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/health`, {
        signal: AbortSignal.timeout(2000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  /**
   * Index raw content into the session collection.
   */
  async index(content: string, source: string): Promise<QmdIndexResult> {
    // Append monotonic counter to source to ensure unique document keys
    // within the session (e.g., multiple "execute:python" calls)
    const key = `${source}:${++this.indexCounter}`;
    const res = await fetch(`${this.baseUrl}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content, source: key, session: this.sessionId }),
      signal: AbortSignal.timeout(5000),
    });

    if (!res.ok) {
      throw new Error(`QMD /index failed: ${res.status} ${await res.text()}`);
    }

    return (await res.json()) as QmdIndexResult;
  }

  /**
   * Search indexed content via QMD's hybrid search pipeline.
   * For each query string, generates both a lex and vec sub-search.
   */
  async search(
    queries: string[],
    options?: { limit?: number; minScore?: number }
  ): Promise<QmdSearchResult[]> {
    const nonEmpty = queries.filter(q => q.trim().length > 0);
    if (nonEmpty.length === 0) return [];

    // Build sub-searches: for each query, do both lex and vec
    const searches: { type: string; query: string }[] = [];
    for (const q of nonEmpty) {
      searches.push({ type: "lex", query: q });
      searches.push({ type: "vec", query: q });
    }

    const res = await fetch(`${this.baseUrl}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        searches,
        limit: options?.limit ?? 10,
        minScore: options?.minScore ?? 0,
        collections: [this.collection],
      }),
      signal: AbortSignal.timeout(10000),
    });

    if (!res.ok) {
      throw new Error(`QMD /query failed: ${res.status} ${await res.text()}`);
    }

    const data = (await res.json()) as { results: QmdSearchResult[] };
    return data.results;
  }

  /**
   * Flush all session content from QMD.
   */
  async flush(): Promise<void> {
    try {
      const res = await fetch(`${this.baseUrl}/flush`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session: this.sessionId }),
        signal: AbortSignal.timeout(2000),
      });
      if (!res.ok) {
        console.error(`QMD /flush failed: ${res.status}`);
      }
    } catch (err) {
      // Best-effort — session collection will be purged by QMD's stale cleanup
      console.error("QMD flush failed:", err);
    }
  }

  getSessionId(): string {
    return this.sessionId;
  }

  getCollection(): string {
    return this.collection;
  }
}
