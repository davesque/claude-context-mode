#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { createRequire } from "node:module";
import { z } from "zod";
import { randomUUID } from "node:crypto";
import { PolyglotExecutor } from "./executor.js";
import { ContentStore, cleanupStaleDBs, type SearchResult } from "./store.js";
import { QmdClient, type QmdSearchResult } from "./qmd-client.js";
import {
  detectRuntimes,
  getRuntimeSummary,
  getAvailableLanguages,
  hasBunRuntime,
} from "./runtime.js";

const VERSION = "0.8.1";
const runtimes = detectRuntimes();
const available = getAvailableLanguages(runtimes);
const server = new McpServer({
  name: "context-mode",
  version: VERSION,
});

const executor = new PolyglotExecutor({
  runtimes,
  projectRoot: process.env.CLAUDE_PROJECT_DIR,
});

// Lazy singleton — no DB overhead unless index/search is used
let _store: ContentStore | null = null;
function getStore(): ContentStore {
  if (!_store) _store = new ContentStore();
  return _store;
}

// QMD hybrid search integration — delegates index/search to QMD daemon when available
const QMD_URL = process.env.QMD_URL || "http://localhost:8181";
const qmdSessionId = randomUUID();
const qmdClient = new QmdClient(QMD_URL, qmdSessionId);
let qmdAvailable = false;

// ─────────────────────────────────────────────────────────
// Session stats — track context consumption per tool
// ─────────────────────────────────────────────────────────

const sessionStats = {
  calls: {} as Record<string, number>,
  bytesReturned: {} as Record<string, number>,
  bytesIndexed: 0,
  bytesSandboxed: 0, // network I/O consumed inside sandbox (never enters context)
  sessionStart: Date.now(),
};

type ToolResult = {
  content: Array<{ type: "text"; text: string }>;
  isError?: boolean;
};

function trackResponse(toolName: string, response: ToolResult): ToolResult {
  const bytes = response.content.reduce(
    (sum, c) => sum + Buffer.byteLength(c.text),
    0,
  );
  sessionStats.calls[toolName] = (sessionStats.calls[toolName] || 0) + 1;
  sessionStats.bytesReturned[toolName] =
    (sessionStats.bytesReturned[toolName] || 0) + bytes;
  return response;
}

function trackIndexed(bytes: number): void {
  sessionStats.bytesIndexed += bytes;
}

// Build description dynamically based on detected runtimes
const langList = available.join(", ");
const bunNote = hasBunRuntime()
  ? " (Bun detected — JS/TS runs 3-5x faster)"
  : "";

// ─────────────────────────────────────────────────────────
// Helper: smart snippet extraction — returns windows around
// matching query terms instead of dumb truncation
//
// When `highlighted` is provided (from FTS5 `highlight()` with
// STX/ETX markers), match positions are derived from the markers.
// This is the authoritative source — FTS5 uses the exact same
// tokenizer that produced the BM25 match, so stemmed variants
// like "configuration" matching query "configure" are found
// correctly. Falls back to indexOf on raw terms when highlighted
// is absent (non-FTS codepath).
// ─────────────────────────────────────────────────────────

const STX = "\x02";
const ETX = "\x03";

/**
 * Parse FTS5 highlight markers to find match positions in the
 * original (marker-free) text. Returns character offsets into the
 * stripped content where each matched token begins.
 */
export function positionsFromHighlight(highlighted: string): number[] {
  const positions: number[] = [];
  let cleanOffset = 0;

  let i = 0;
  while (i < highlighted.length) {
    if (highlighted[i] === STX) {
      // Record position of this match in the clean text
      positions.push(cleanOffset);
      i++; // skip STX
      // Advance through matched text until ETX
      while (i < highlighted.length && highlighted[i] !== ETX) {
        cleanOffset++;
        i++;
      }
      if (i < highlighted.length) i++; // skip ETX
    } else {
      cleanOffset++;
      i++;
    }
  }

  return positions;
}

/** Strip STX/ETX markers to recover original content. */
function stripMarkers(highlighted: string): string {
  return highlighted.replaceAll(STX, "").replaceAll(ETX, "");
}

export function extractSnippet(
  content: string,
  query: string,
  maxLen = 1500,
  highlighted?: string,
): string {
  if (content.length <= maxLen) return content;

  // Derive match positions from FTS5 highlight markers when available
  const positions: number[] = [];

  if (highlighted) {
    for (const pos of positionsFromHighlight(highlighted)) {
      positions.push(pos);
    }
  }

  // Fallback: indexOf on raw query terms (non-FTS codepath)
  if (positions.length === 0) {
    const terms = query
      .toLowerCase()
      .split(/\s+/)
      .filter((t) => t.length > 2);
    const lower = content.toLowerCase();

    for (const term of terms) {
      let idx = lower.indexOf(term);
      while (idx !== -1) {
        positions.push(idx);
        idx = lower.indexOf(term, idx + 1);
      }
    }
  }

  // No matches at all — return prefix
  if (positions.length === 0) {
    return content.slice(0, maxLen) + "\n…";
  }

  // Sort positions, merge overlapping windows
  positions.sort((a, b) => a - b);
  const WINDOW = 300;
  const windows: Array<[number, number]> = [];

  for (const pos of positions) {
    const start = Math.max(0, pos - WINDOW);
    const end = Math.min(content.length, pos + WINDOW);
    if (windows.length > 0 && start <= windows[windows.length - 1][1]) {
      windows[windows.length - 1][1] = end;
    } else {
      windows.push([start, end]);
    }
  }

  // Collect windows until maxLen
  const parts: string[] = [];
  let total = 0;
  for (const [start, end] of windows) {
    if (total >= maxLen) break;
    const part = content.slice(start, Math.min(end, start + (maxLen - total)));
    parts.push(
      (start > 0 ? "…" : "") + part + (end < content.length ? "…" : ""),
    );
    total += part.length;
  }

  return parts.join("\n\n");
}

// ─────────────────────────────────────────────────────────
// Tool: execute
// ─────────────────────────────────────────────────────────

server.registerTool(
  "execute",
  {
    title: "Execute Code",
    description: `Execute code in a sandboxed subprocess. Only stdout enters context — raw data stays in the subprocess. Use instead of bash/cat when output would exceed 20 lines.${bunNote} Available: ${langList}.\n\nPREFER THIS OVER BASH for: API calls (gh, curl, aws), test runners (npm test, pytest), git queries (git log, git diff), data processing, and ANY CLI command that may produce large output. Bash should only be used for file mutations, git writes, and navigation.`,
    inputSchema: z.object({
      language: z
        .enum([
          "javascript",
          "typescript",
          "python",
          "shell",
          "ruby",
          "go",
          "rust",
          "php",
          "perl",
          "r",
          "elixir",
        ])
        .describe("Runtime language"),
      code: z
        .string()
        .describe(
          "Source code to execute. Use console.log (JS/TS), print (Python/Ruby/Perl/R), echo (Shell), echo (PHP), fmt.Println (Go), or IO.puts (Elixir) to output a summary to context.",
        ),
      timeout: z
        .number()
        .optional()
        .default(30000)
        .describe("Max execution time in ms"),
      intent: z
        .string()
        .optional()
        .describe(
          "What you're looking for in the output. When provided and output is large (>5KB), " +
          "indexes output into hybrid search backend and returns section titles + previews — not full content. " +
          "Use search(queries: [...]) to retrieve specific sections. Example: 'failing tests', 'HTTP 500 errors'." +
          "\n\nTIP: Use specific technical terms, not just concepts. Check 'Searchable terms' in the response for available vocabulary.",
        ),
    }),
  },
  async ({ language, code, timeout, intent }) => {
    try {
      // For JS/TS: wrap in async IIFE with fetch interceptor to track network bytes
      let instrumentedCode = code;
      if (language === "javascript" || language === "typescript") {
        instrumentedCode = `
let __cm_net=0;const __cm_f=globalThis.fetch;
globalThis.fetch=async(...a)=>{const r=await __cm_f(...a);
try{const cl=r.clone();const b=await cl.arrayBuffer();__cm_net+=b.byteLength}catch{}
return r};
async function __cm_main(){
${code}
}
__cm_main().catch(e=>{console.error(e);process.exitCode=1}).finally(()=>{
if(__cm_net>0)process.stderr.write('__CM_NET__:'+__cm_net+'\\n');
});`;
      }
      const result = await executor.execute({ language, code: instrumentedCode, timeout });

      // Parse sandbox network metrics from stderr
      const netMatch = result.stderr?.match(/__CM_NET__:(\d+)/);
      if (netMatch) {
        sessionStats.bytesSandboxed += parseInt(netMatch[1]);
        // Clean the metric line from stderr
        result.stderr = result.stderr.replace(/\n?__CM_NET__:\d+\n?/g, "");
      }

      if (result.timedOut) {
        return trackResponse("execute", {
          content: [
            {
              type: "text" as const,
              text: `Execution timed out after ${timeout}ms\n\nPartial stdout:\n${result.stdout}\n\nstderr:\n${result.stderr}`,
            },
          ],
          isError: true,
        });
      }

      if (result.exitCode !== 0) {
        const output = `Exit code: ${result.exitCode}\n\nstdout:\n${result.stdout}\n\nstderr:\n${result.stderr}`;
        if (intent && intent.trim().length > 0 && Buffer.byteLength(output) > INTENT_SEARCH_THRESHOLD) {
          trackIndexed(Buffer.byteLength(output));
          return trackResponse("execute", {
            content: [
              { type: "text" as const, text: await intentSearch(output, intent, `execute:${language}:error`) },
            ],
            isError: true,
          });
        }
        return trackResponse("execute", {
          content: [
            { type: "text" as const, text: output },
          ],
          isError: true,
        });
      }

      const stdout = result.stdout || "(no output)";

      // Intent-driven search: if intent provided and output is large enough
      if (intent && intent.trim().length > 0 && Buffer.byteLength(stdout) > INTENT_SEARCH_THRESHOLD) {
        trackIndexed(Buffer.byteLength(stdout));
        return trackResponse("execute", {
          content: [
            { type: "text" as const, text: await intentSearch(stdout, intent, `execute:${language}`) },
          ],
        });
      }

      return trackResponse("execute", {
        content: [
          { type: "text" as const, text: stdout },
        ],
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      return trackResponse("execute", {
        content: [
          { type: "text" as const, text: `Runtime error: ${message}` },
        ],
        isError: true,
      });
    }
  },
);

// ─────────────────────────────────────────────────────────
// Helper: intent-driven search on execution output
// ─────────────────────────────────────────────────────────

const INTENT_SEARCH_THRESHOLD = 5_000; // bytes — ~80-100 lines

async function intentSearch(
  stdout: string,
  intent: string,
  source: string,
  maxResults: number = 5,
): Promise<string> {
  const totalLines = stdout.split("\n").length;
  const totalBytes = Buffer.byteLength(stdout);

  // QMD path: index + search via hybrid pipeline
  if (qmdAvailable) {
    try {
      const indexResult = await qmdClient.index(stdout, source);
      let searchResults = await qmdClient.search([intent], { limit: maxResults });

      // Scope to just-indexed document (source substring matches "source:N" keys)
      const sourceLower = source.toLowerCase();
      searchResults = searchResults.filter(r =>
        r.file.toLowerCase().includes(sourceLower)
      );

      if (searchResults.length === 0) {
        return [
          `Indexed "${source}" into QMD.`,
          `No sections matched intent "${intent}" in ${totalLines}-line output (${(totalBytes / 1024).toFixed(1)}KB).`,
          "",
          "Use search() to explore the indexed content.",
        ].join("\n");
      }

      const lines = [
        `Indexed "${source}" into QMD.`,
        `${searchResults.length} sections matched "${intent}" (${totalLines} lines, ${(totalBytes / 1024).toFixed(1)}KB):`,
        "",
      ];

      for (const r of searchResults) {
        const preview = r.snippet.split("\n")[0].slice(0, 120);
        lines.push(`  - ${r.title}: ${preview}`);
      }

      lines.push("");
      lines.push("Use search(queries: [...]) to retrieve full content of any section.");
      return lines.join("\n");
    } catch (err) {
      console.error("QMD intentSearch failed, falling back to local:", err);
    }
  }

  // Fallback: local FTS5 store
  const persistent = getStore();
  const indexed = persistent.indexPlainText(stdout, source);

  let results = persistent.searchWithFallback(intent, maxResults, source);

  const distinctiveTerms = persistent.getDistinctiveTerms(indexed.sourceId);

  if (results.length === 0) {
    const lines = [
      `Indexed ${indexed.totalChunks} sections from "${source}" into knowledge base.`,
      `No sections matched intent "${intent}" in ${totalLines}-line output (${(totalBytes / 1024).toFixed(1)}KB).`,
    ];
    if (distinctiveTerms.length > 0) {
      lines.push("");
      lines.push(`Searchable terms: ${distinctiveTerms.join(", ")}`);
    }
    lines.push("");
    lines.push("Use search() to explore the indexed content.");
    return lines.join("\n");
  }

  const lines = [
    `Indexed ${indexed.totalChunks} sections from "${source}" into knowledge base.`,
    `${results.length} sections matched "${intent}" (${totalLines} lines, ${(totalBytes / 1024).toFixed(1)}KB):`,
    "",
  ];

  for (const r of results) {
    const preview = r.content.split("\n")[0].slice(0, 120);
    lines.push(`  - ${r.title}: ${preview}`);
  }

  if (distinctiveTerms.length > 0) {
    lines.push("");
    lines.push(`Searchable terms: ${distinctiveTerms.join(", ")}`);
  }

  lines.push("");
  lines.push("Use search(queries: [...]) to retrieve full content of any section.");

  return lines.join("\n");
}

// ─────────────────────────────────────────────────────────
// Tool: execute_file
// ─────────────────────────────────────────────────────────

server.registerTool(
  "execute_file",
  {
    title: "Execute File Processing",
    description:
      "Read a file and process it without loading contents into context. The file is read into a FILE_CONTENT variable inside the sandbox. Only your printed summary enters context.\n\nPREFER THIS OVER Read/cat for: log files, data files (CSV, JSON, XML), large source files for analysis, and any file where you need to extract specific information rather than read the entire content.",
    inputSchema: z.object({
      path: z
        .string()
        .describe("Absolute file path or relative to project root"),
      language: z
        .enum([
          "javascript",
          "typescript",
          "python",
          "shell",
          "ruby",
          "go",
          "rust",
          "php",
          "perl",
          "r",
          "elixir",
        ])
        .describe("Runtime language"),
      code: z
        .string()
        .describe(
          "Code to process FILE_CONTENT (file_content in Elixir). Print summary via console.log/print/echo/IO.puts.",
        ),
      timeout: z
        .number()
        .optional()
        .default(30000)
        .describe("Max execution time in ms"),
      intent: z
        .string()
        .optional()
        .describe(
          "What you're looking for in the output. When provided and output is large (>5KB), " +
          "returns only matching sections via hybrid search instead of truncated output.",
        ),
    }),
  },
  async ({ path, language, code, timeout, intent }) => {
    try {
      const result = await executor.executeFile({
        path,
        language,
        code,
        timeout,
      });

      if (result.timedOut) {
        return trackResponse("execute_file", {
          content: [
            {
              type: "text" as const,
              text: `Timed out processing ${path} after ${timeout}ms`,
            },
          ],
          isError: true,
        });
      }

      if (result.exitCode !== 0) {
        const output = `Error processing ${path} (exit ${result.exitCode}):\n${result.stderr || result.stdout}`;
        if (intent && intent.trim().length > 0 && Buffer.byteLength(output) > INTENT_SEARCH_THRESHOLD) {
          trackIndexed(Buffer.byteLength(output));
          return trackResponse("execute_file", {
            content: [
              { type: "text" as const, text: await intentSearch(output, intent, `file:${path}:error`) },
            ],
            isError: true,
          });
        }
        return trackResponse("execute_file", {
          content: [
            { type: "text" as const, text: output },
          ],
          isError: true,
        });
      }

      const stdout = result.stdout || "(no output)";

      if (intent && intent.trim().length > 0 && Buffer.byteLength(stdout) > INTENT_SEARCH_THRESHOLD) {
        trackIndexed(Buffer.byteLength(stdout));
        return trackResponse("execute_file", {
          content: [
            { type: "text" as const, text: await intentSearch(stdout, intent, `file:${path}`) },
          ],
        });
      }

      return trackResponse("execute_file", {
        content: [
          { type: "text" as const, text: stdout },
        ],
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      return trackResponse("execute_file", {
        content: [
          { type: "text" as const, text: `Runtime error: ${message}` },
        ],
        isError: true,
      });
    }
  },
);

// ─────────────────────────────────────────────────────────
// Tool: index
// ─────────────────────────────────────────────────────────

server.registerTool(
  "index",
  {
    title: "Index Content",
    description:
      "Index documentation or knowledge content into a searchable hybrid knowledge base (BM25 + semantic). " +
      "When QMD is running, uses vector embeddings and reranking for higher-quality results; otherwise falls back to local FTS5. " +
      "Chunks markdown by headings (keeping code blocks intact). " +
      "The full content does NOT stay in context — only a brief summary is returned.\n\n" +
      "WHEN TO USE:\n" +
      "- Documentation from Context7, Skills, or MCP tools (API docs, framework guides, code examples)\n" +
      "- API references (endpoint details, parameter specs, response schemas)\n" +
      "- MCP tools/list output (exact tool signatures and descriptions)\n" +
      "- Skill prompts and instructions that are too large for context\n" +
      "- README files, migration guides, changelog entries\n" +
      "- Any content with code examples you may need to reference precisely\n\n" +
      "After indexing, use 'search' to retrieve specific sections on-demand.\n" +
      "Do NOT use for: log files, test output, CSV, build output — use 'execute_file' for those.",
    inputSchema: z.object({
      content: z
        .string()
        .optional()
        .describe(
          "Raw text/markdown to index. Provide this OR path, not both.",
        ),
      path: z
        .string()
        .optional()
        .describe(
          "File path to read and index (content never enters context). Provide this OR content.",
        ),
      source: z
        .string()
        .optional()
        .describe(
          "Label for the indexed content (e.g., 'Context7: React useEffect', 'Skill: frontend-design')",
        ),
    }),
  },
  async ({ content, path, source }) => {
    if (!content && !path) {
      return trackResponse("index", {
        content: [
          {
            type: "text" as const,
            text: "Error: Either content or path must be provided",
          },
        ],
        isError: true,
      });
    }

    try {
      // Resolve content from file path if needed
      let indexContent = content;
      if (!indexContent && path) {
        const fs = await import("fs");
        indexContent = fs.readFileSync(path, "utf-8");
      }
      if (!indexContent) {
        return trackResponse("index", {
          content: [{ type: "text" as const, text: "Error: No content to index" }],
          isError: true,
        });
      }

      trackIndexed(Buffer.byteLength(indexContent));
      const label = source ?? path ?? "inline";

      // QMD path
      if (qmdAvailable) {
        try {
          const result = await qmdClient.index(indexContent, label);
          return trackResponse("index", {
            content: [
              {
                type: "text" as const,
                text: `Indexed from: ${label}\nUse search(queries: ["..."]) to query this content.`,
              },
            ],
          });
        } catch (err) {
          console.error("QMD index failed, falling back to local:", err);
        }
      }

      // Fallback: local FTS5 store
      const store = getStore();
      const result = store.index({ content: indexContent, source: label });

      return trackResponse("index", {
        content: [
          {
            type: "text" as const,
            text: `Indexed ${result.totalChunks} sections (${result.codeChunks} with code) from: ${result.label}\nUse search(queries: ["..."]) to query this content. Use source: "${result.label}" to scope results.`,
          },
        ],
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      return trackResponse("index", {
        content: [
          { type: "text" as const, text: `Index error: ${message}` },
        ],
        isError: true,
      });
    }
  },
);

// ─────────────────────────────────────────────────────────
// Tool: search — progressive throttling
// ─────────────────────────────────────────────────────────

// Track search calls per 60-second window for progressive throttling
let searchCallCount = 0;
let searchWindowStart = Date.now();
const SEARCH_WINDOW_MS = 60_000;
const SEARCH_MAX_RESULTS_AFTER = 3; // after 3 calls: 1 result per query
const SEARCH_BLOCK_AFTER = 8; // after 8 calls: refuse, demand batching

server.registerTool(
  "search",
  {
    title: "Search Indexed Content",
    description:
      "Search indexed content. Pass ALL search questions as queries array in ONE call.\n\n" +
      "TIPS: Natural language queries work well (e.g. 'how does authentication work'). Keyword queries also work. Use 'source' to scope results.",
    inputSchema: z.object({
      queries: z
        .array(z.string())
        .optional()
        .describe("Array of search queries. Batch ALL questions in one call."),
      limit: z
        .number()
        .optional()
        .default(3)
        .describe("Results per query (default: 3)"),
      source: z
        .string()
        .optional()
        .describe("Filter to a specific indexed source (partial match)."),
    }),
  },
  async (params) => {
    try {
      const raw = params as Record<string, unknown>;

      // Normalize: accept both query (string) and queries (array)
      const queryList: string[] = [];
      if (Array.isArray(raw.queries) && raw.queries.length > 0) {
        queryList.push(...(raw.queries as string[]));
      } else if (typeof raw.query === "string" && raw.query.length > 0) {
        queryList.push(raw.query as string);
      }

      if (queryList.length === 0) {
        return trackResponse("search", {
          content: [{ type: "text" as const, text: "Error: provide query or queries." }],
          isError: true,
        });
      }

      const { limit = 3, source } = params as { limit?: number; source?: string };

      // QMD path: hybrid search (BM25 + vector + RRF + reranking).
      // No progressive throttling — QMD's reranking produces higher-quality
      // results per query, and the per-query round-trip already discourages
      // excessive calls more naturally than the local FTS5 path.
      if (qmdAvailable) {
        try {
          const MAX_TOTAL = 40 * 1024;
          let totalSize = 0;
          const sections: string[] = [];

          // Issue one search per query for proper attribution
          for (const q of queryList) {
            if (totalSize > MAX_TOTAL) {
              sections.push(`## ${q}\n(output cap reached)\n`);
              continue;
            }

            let qResults = await qmdClient.search([q], { limit });

            // Client-side source filtering (QMD scopes to session collection,
            // but source filters within it by document path)
            if (source && qResults.length > 0) {
              const sourceLower = source.toLowerCase();
              qResults = qResults.filter(r =>
                r.file.toLowerCase().includes(sourceLower)
              );
            }

            if (qResults.length === 0) {
              sections.push(`## ${q}\nNo results found.`);
              continue;
            }

            const formatted = qResults
              .map(r => {
                const header = `--- [${r.file}] ---`;
                const heading = `### ${r.title}`;
                return `${header}\n${heading}\n\n${r.snippet}`;
              })
              .join("\n\n");

            sections.push(`## ${q}\n\n${formatted}`);
            totalSize += formatted.length;
          }

          const output = sections.join("\n\n---\n\n");
          return trackResponse("search", {
            content: [{ type: "text" as const, text: output || "No results found." }],
          });
        } catch (err) {
          console.error("QMD search failed, falling back to local:", err);
        }
      }

      // Fallback: local FTS5 store with progressive throttling
      const store = getStore();

      const now = Date.now();
      if (now - searchWindowStart > SEARCH_WINDOW_MS) {
        searchCallCount = 0;
        searchWindowStart = now;
      }
      searchCallCount++;

      if (searchCallCount > SEARCH_BLOCK_AFTER) {
        return trackResponse("search", {
          content: [{
            type: "text" as const,
            text: `BLOCKED: ${searchCallCount} search calls in ${Math.round((now - searchWindowStart) / 1000)}s. ` +
              "You're flooding context. STOP making individual search calls. " +
              "Use batch_execute(commands, queries) for your next research step.",
          }],
          isError: true,
        });
      }

      const effectiveLimit = searchCallCount > SEARCH_MAX_RESULTS_AFTER
        ? 1
        : Math.min(limit, 2);

      const MAX_TOTAL = 40 * 1024;
      let totalSize = 0;
      const sections: string[] = [];

      for (const q of queryList) {
        if (totalSize > MAX_TOTAL) {
          sections.push(`## ${q}\n(output cap reached)\n`);
          continue;
        }

        const results = store.searchWithFallback(q, effectiveLimit, source);

        if (results.length === 0) {
          sections.push(`## ${q}\nNo results found.`);
          continue;
        }

        const formatted = results
          .map((r) => {
            const header = `--- [${r.source}] ---`;
            const heading = `### ${r.title}`;
            const snippet = extractSnippet(r.content, q, 1500, r.highlighted);
            return `${header}\n${heading}\n\n${snippet}`;
          })
          .join("\n\n");

        sections.push(`## ${q}\n\n${formatted}`);
        totalSize += formatted.length;
      }

      let output = sections.join("\n\n---\n\n");

      if (searchCallCount >= SEARCH_MAX_RESULTS_AFTER) {
        output += `\n\n⚠ search call #${searchCallCount}/${SEARCH_BLOCK_AFTER} in this window. ` +
          `Results limited to ${effectiveLimit}/query. ` +
          `Batch queries: search(queries: ["q1","q2","q3"]) or use batch_execute.`;
      }

      if (output.trim().length === 0) {
        const sources = store.listSources();
        const sourceList = sources.length > 0
          ? `\nIndexed sources: ${sources.map((s) => `"${s.label}" (${s.chunkCount} sections)`).join(", ")}`
          : "";
        return trackResponse("search", {
          content: [{ type: "text" as const, text: `No results found.${sourceList}` }],
        });
      }

      return trackResponse("search", {
        content: [{ type: "text" as const, text: output }],
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      return trackResponse("search", {
        content: [{ type: "text" as const, text: `Search error: ${message}` }],
        isError: true,
      });
    }
  },
);

// ─────────────────────────────────────────────────────────
// Turndown path resolution (external dep, like better-sqlite3)
// ─────────────────────────────────────────────────────────

let _turndownPath: string | null = null;
let _gfmPluginPath: string | null = null;

function resolveTurndownPath(): string {
  if (!_turndownPath) {
    const require = createRequire(import.meta.url);
    _turndownPath = require.resolve("turndown");
  }
  return _turndownPath;
}

function resolveGfmPluginPath(): string {
  if (!_gfmPluginPath) {
    const require = createRequire(import.meta.url);
    _gfmPluginPath = require.resolve("turndown-plugin-gfm");
  }
  return _gfmPluginPath;
}

// ─────────────────────────────────────────────────────────
// Tool: fetch_and_index
// ─────────────────────────────────────────────────────────

function buildFetchCode(url: string): string {
  const turndownPath = JSON.stringify(resolveTurndownPath());
  const gfmPath = JSON.stringify(resolveGfmPluginPath());
  return `
const TurndownService = require(${turndownPath});
const { gfm } = require(${gfmPath});
const url = ${JSON.stringify(url)};

async function main() {
  const resp = await fetch(url);
  if (!resp.ok) { console.error("HTTP " + resp.status); process.exit(1); }
  const html = await resp.text();

  const td = new TurndownService({ headingStyle: 'atx', codeBlockStyle: 'fenced' });
  td.use(gfm);
  td.remove(['script', 'style', 'nav', 'header', 'footer', 'noscript']);
  console.log(td.turndown(html));
}
main();
`;
}

server.registerTool(
  "fetch_and_index",
  {
    title: "Fetch & Index URL",
    description:
      "Fetches URL content, converts HTML to markdown, indexes into hybrid search backend, " +
      "and returns a ~3KB preview. Full content stays in sandbox — use search() for deeper lookups.\n\n" +
      "Better than WebFetch: preview is immediate, full content is searchable, raw HTML never enters context.",
    inputSchema: z.object({
      url: z.string().describe("The URL to fetch and index"),
      source: z
        .string()
        .optional()
        .describe(
          "Label for the indexed content (e.g., 'React useEffect docs', 'Supabase Auth API')",
        ),
    }),
  },
  async ({ url, source }) => {
    try {
      // Execute fetch inside subprocess — raw HTML never enters context
      const fetchCode = buildFetchCode(url);
      const result = await executor.execute({
        language: "javascript",
        code: fetchCode,
        timeout: 30_000,
      });

      if (result.exitCode !== 0) {
        return trackResponse("fetch_and_index", {
          content: [
            {
              type: "text" as const,
              text: `Failed to fetch ${url}: ${result.stderr || result.stdout}`,
            },
          ],
          isError: true,
        });
      }

      if (!result.stdout || result.stdout.trim().length === 0) {
        return trackResponse("fetch_and_index", {
          content: [
            {
              type: "text" as const,
              text: `Fetched ${url} but got empty content after HTML conversion`,
            },
          ],
          isError: true,
        });
      }

      const markdown = result.stdout.trim();
      trackIndexed(Buffer.byteLength(markdown));
      const label = source ?? url;

      // Build preview — first ~3KB of markdown for immediate use
      const PREVIEW_LIMIT = 3072;
      const preview = markdown.length > PREVIEW_LIMIT
        ? markdown.slice(0, PREVIEW_LIMIT) + "\n\n…[truncated — use search() for full content]"
        : markdown;
      const totalKB = (Buffer.byteLength(markdown) / 1024).toFixed(1);

      // QMD path
      if (qmdAvailable) {
        try {
          const indexResult = await qmdClient.index(markdown, label);
          const text = [
            `Fetched and indexed (${totalKB}KB) from: ${label}`,
            `Full content indexed — use search(queries: [...]) for specific lookups.`,
            "",
            "---",
            "",
            preview,
          ].join("\n");

          return trackResponse("fetch_and_index", {
            content: [{ type: "text" as const, text }],
          });
        } catch (err) {
          console.error("QMD index failed in fetch_and_index, falling back to local:", err);
        }
      }

      // Fallback: local FTS5 store
      const store = getStore();
      const indexed = store.index({ content: markdown, source: label });

      const text = [
        `Fetched and indexed **${indexed.totalChunks} sections** (${totalKB}KB) from: ${indexed.label}`,
        `Full content indexed in sandbox — use search(queries: [...], source: "${indexed.label}") for specific lookups.`,
        "",
        "---",
        "",
        preview,
      ].join("\n");

      return trackResponse("fetch_and_index", {
        content: [{ type: "text" as const, text }],
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      return trackResponse("fetch_and_index", {
        content: [
          { type: "text" as const, text: `Fetch error: ${message}` },
        ],
        isError: true,
      });
    }
  },
);

// ─────────────────────────────────────────────────────────
// Tool: batch_execute
// ─────────────────────────────────────────────────────────

server.registerTool(
  "batch_execute",
  {
    title: "Batch Execute & Search",
    description:
      "Execute multiple commands in ONE call, auto-index all output, and search with multiple queries. " +
      "Returns search results directly — no follow-up calls needed.\n\n" +
      "THIS IS THE PRIMARY TOOL. Use this instead of multiple execute() calls.\n\n" +
      "One batch_execute call replaces 30+ execute calls + 10+ search calls.\n" +
      "Provide all commands to run and all queries to search — everything happens in one round trip.",
    inputSchema: z.object({
      commands: z
        .array(
          z.object({
            label: z
              .string()
              .describe(
                "Section header for this command's output (e.g., 'README', 'Package.json', 'Source Tree')",
              ),
            command: z
              .string()
              .describe("Shell command to execute"),
          }),
        )
        .min(1)
        .describe(
          "Commands to execute as a batch. Each runs sequentially, output is labeled with the section header.",
        ),
      queries: z
        .array(z.string())
        .min(1)
        .describe(
          "Search queries to extract information from indexed output. Use 5-8 comprehensive queries. " +
          "Each returns top 5 matching sections with full content. " +
          "This is your ONLY chance — put ALL your questions here. No follow-up calls needed.",
        ),
      timeout: z
        .number()
        .optional()
        .default(60000)
        .describe("Max execution time in ms (default: 60s)"),
    }),
  },
  async ({ commands, queries, timeout }) => {
    try {
      // Build batch script with markdown section headers for proper chunking
      const script = commands
        .map((c) => {
          const safeLabel = c.label.replace(/'/g, "'\\''");
          return `echo '# ${safeLabel}'\necho ''\n${c.command} 2>&1\necho ''`;
        })
        .join("\n");

      const result = await executor.execute({
        language: "shell",
        code: script,
        timeout,
      });

      if (result.timedOut) {
        return trackResponse("batch_execute", {
          content: [
            {
              type: "text" as const,
              text: `Batch timed out after ${timeout}ms. Partial output:\n${result.stdout?.slice(0, 2000) || "(none)"}`,
            },
          ],
          isError: true,
        });
      }

      const stdout = result.stdout || "(no output)";
      const totalBytes = Buffer.byteLength(stdout);
      const totalLines = stdout.split("\n").length;

      trackIndexed(totalBytes);

      const source = `batch:${commands
        .map((c) => c.label)
        .join(",")
        .slice(0, 80)}`;

      // QMD path: index + search via hybrid pipeline
      if (qmdAvailable) {
        try {
          const indexResult = await qmdClient.index(stdout, source);

          // Scope searches to just-indexed document
          const sourceLower = source.toLowerCase();

          const MAX_OUTPUT = 80 * 1024;
          const queryResults: string[] = [];
          let outputSize = 0;

          // Issue one search per query for proper attribution
          for (const query of queries) {
            if (outputSize > MAX_OUTPUT) {
              queryResults.push(`## ${query}\n(output cap reached)\n`);
              continue;
            }

            const qResults = (await qmdClient.search([query], { limit: 3 }))
              .filter(r => r.file.toLowerCase().includes(sourceLower));

            queryResults.push(`## ${query}`);
            queryResults.push("");
            if (qResults.length > 0) {
              for (const r of qResults) {
                queryResults.push(`### ${r.title}`);
                queryResults.push(r.snippet);
                queryResults.push("");
                outputSize += r.snippet.length + r.title.length;
              }
            } else {
              queryResults.push("No matching sections found.");
              queryResults.push("");
            }
          }

          const output = [
            `Executed ${commands.length} commands (${totalLines} lines, ${(totalBytes / 1024).toFixed(1)}KB). ` +
              `Indexed into QMD. Searched ${queries.length} queries.`,
            "",
            ...queryResults,
          ].join("\n");

          return trackResponse("batch_execute", {
            content: [{ type: "text" as const, text: output }],
          });
        } catch (err) {
          console.error("QMD batch_execute failed, falling back to local:", err);
        }
      }

      // Fallback: local FTS5 store
      const store = getStore();
      const indexed = store.index({ content: stdout, source });

      const allSections = store.getChunksBySource(indexed.sourceId);
      const inventory: string[] = ["## Indexed Sections", ""];
      const sectionTitles: string[] = [];
      for (const s of allSections) {
        const bytes = Buffer.byteLength(s.content);
        inventory.push(`- ${s.title} (${(bytes / 1024).toFixed(1)}KB)`);
        sectionTitles.push(s.title);
      }

      const MAX_OUTPUT = 80 * 1024;
      const queryResults: string[] = [];
      let outputSize = 0;

      for (const query of queries) {
        if (outputSize > MAX_OUTPUT) {
          queryResults.push(`## ${query}\n(output cap reached — use search(queries: ["${query}"]) for details)\n`);
          continue;
        }

        let results = store.searchWithFallback(query, 3, source);

        if (results.length === 0) {
          results = store.searchWithFallback(query, 3);
        }

        queryResults.push(`## ${query}`);
        queryResults.push("");
        if (results.length > 0) {
          for (const r of results) {
            const snippet = extractSnippet(r.content, query, 1500, r.highlighted);
            queryResults.push(`### ${r.title}`);
            queryResults.push(snippet);
            queryResults.push("");
            outputSize += snippet.length + r.title.length;
          }
        } else {
          queryResults.push("No matching sections found.");
          queryResults.push("");
        }
      }

      const distinctiveTerms = store.getDistinctiveTerms
        ? store.getDistinctiveTerms(indexed.sourceId)
        : [];

      const output = [
        `Executed ${commands.length} commands (${totalLines} lines, ${(totalBytes / 1024).toFixed(1)}KB). ` +
          `Indexed ${indexed.totalChunks} sections. Searched ${queries.length} queries.`,
        "",
        ...inventory,
        "",
        ...queryResults,
        distinctiveTerms.length > 0
          ? `\nSearchable terms for follow-up: ${distinctiveTerms.join(", ")}`
          : "",
      ].join("\n");

      return trackResponse("batch_execute", {
        content: [{ type: "text" as const, text: output }],
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      return trackResponse("batch_execute", {
        content: [
          {
            type: "text" as const,
            text: `Batch execution error: ${message}`,
          },
        ],
        isError: true,
      });
    }
  },
);

// ─────────────────────────────────────────────────────────
// Tool: stats
// ─────────────────────────────────────────────────────────

server.registerTool(
  "stats",
  {
    title: "Session Statistics",
    description:
      "Returns context consumption statistics for the current session. " +
      "Shows total bytes returned to context, breakdown by tool, call counts, " +
      "estimated token usage, and context savings ratio.",
    inputSchema: z.object({}),
  },
  async () => {
    const totalBytesReturned = Object.values(sessionStats.bytesReturned).reduce(
      (sum, b) => sum + b,
      0,
    );
    const totalCalls = Object.values(sessionStats.calls).reduce(
      (sum, c) => sum + c,
      0,
    );
    const uptimeMs = Date.now() - sessionStats.sessionStart;
    const uptimeMin = (uptimeMs / 60_000).toFixed(1);

    // Total data kept out of context = indexed (QMD/FTS5) + sandboxed (network I/O inside sandbox)
    const keptOut = sessionStats.bytesIndexed + sessionStats.bytesSandboxed;
    const totalProcessed = keptOut + totalBytesReturned;
    const savingsRatio = totalProcessed / Math.max(totalBytesReturned, 1);
    const reductionPct = totalProcessed > 0
      ? ((1 - totalBytesReturned / totalProcessed) * 100).toFixed(0)
      : "0";

    const kb = (b: number) => {
      if (b >= 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)}MB`;
      return `${(b / 1024).toFixed(1)}KB`;
    };

    // ── Summary table ──
    const lines: string[] = [
      `## context-mode session stats`,
      "",
      `| Metric | Value |`,
      `|--------|------:|`,
      `| Session | ${uptimeMin} min |`,
      `| Tool calls | ${totalCalls} |`,
      `| Total data processed | **${kb(totalProcessed)}** |`,
      `| Kept in sandbox | **${kb(keptOut)}** |`,
      `| Entered context | ${kb(totalBytesReturned)} |`,
      `| Tokens consumed | ~${Math.round(totalBytesReturned / 4).toLocaleString()} |`,
      `| **Context savings** | **${savingsRatio.toFixed(1)}x (${reductionPct}% reduction)** |`,
    ];

    // ── Per-tool table ──
    const toolNames = new Set([
      ...Object.keys(sessionStats.calls),
      ...Object.keys(sessionStats.bytesReturned),
    ]);

    if (toolNames.size > 0) {
      lines.push(
        "",
        `| Tool | Calls | Context | Tokens |`,
        `|------|------:|--------:|-------:|`,
      );
      for (const tool of Array.from(toolNames).sort()) {
        const calls = sessionStats.calls[tool] || 0;
        const bytes = sessionStats.bytesReturned[tool] || 0;
        const tokens = Math.round(bytes / 4);
        lines.push(`| ${tool} | ${calls} | ${kb(bytes)} | ~${tokens.toLocaleString()} |`);
      }
      lines.push(`| **Total** | **${totalCalls}** | **${kb(totalBytesReturned)}** | **~${Math.round(totalBytesReturned / 4).toLocaleString()}** |`);
    }

    // ── DevRel summary ──
    const tokensSaved = Math.round(keptOut / 4);
    if (totalCalls === 0) {
      lines.push("", "> No context-mode calls this session. Use `batch_execute` to run commands, `fetch_and_index` for URLs, or `execute` to process data in sandbox.");
    } else if (keptOut === 0) {
      lines.push("", `> context-mode handled **${totalCalls}** tool calls. All outputs were compact enough to enter context directly. Process larger data or batch multiple commands for bigger savings.`);
    } else {
      lines.push("", `> Without context-mode, **${kb(totalProcessed)}** of raw tool output would flood your context window. Instead, **${kb(keptOut)}** (${reductionPct}%) stayed in sandbox — saving **~${tokensSaved.toLocaleString()} tokens** of context space.`);
    }

    const text = lines.join("\n");
    return trackResponse("stats", {
      content: [{ type: "text" as const, text }],
    });
  },
);

// ─────────────────────────────────────────────────────────
// Server startup
// ─────────────────────────────────────────────────────────

async function main() {
  // Clean up stale DB files from previous sessions
  const cleaned = cleanupStaleDBs();
  if (cleaned > 0) {
    console.error(`Cleaned up ${cleaned} stale DB file(s) from previous sessions`);
  }

  // Check QMD availability
  qmdAvailable = await qmdClient.isAvailable();
  if (qmdAvailable) {
    console.error(`QMD hybrid search available at ${QMD_URL} (session: ${qmdSessionId})`);
  } else {
    console.error(`QMD not available at ${QMD_URL} — using local FTS5 search`);
  }

  // Clean up on shutdown: flush QMD session + local DB
  const shutdown = async () => {
    if (qmdAvailable) {
      try { await qmdClient.flush(); } catch {}
    }
    if (_store) _store.cleanup();
  };
  // 'exit' handler is sync — only local cleanup works here
  process.on("exit", () => { if (_store) _store.cleanup(); });
  process.on("SIGINT", async () => { await shutdown(); process.exit(0); });
  process.on("SIGTERM", async () => { await shutdown(); process.exit(0); });

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`Context Mode MCP server v${VERSION} running on stdio`);
  console.error(`Detected runtimes:\n${getRuntimeSummary(runtimes)}`);
  if (!hasBunRuntime()) {
    console.error(
      "\nPerformance tip: Install Bun for 3-5x faster JS/TS execution",
    );
    console.error("  curl -fsSL https://bun.sh/install | bash");
  }
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
