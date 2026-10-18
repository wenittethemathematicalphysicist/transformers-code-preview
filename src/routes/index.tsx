import { createFileRoute } from "@tanstack/react-router";
import { useRef, useState } from "react";
import { EXAMPLES } from "@/lib/examples";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Transformers Playground — run models in your browser" },
      { name: "description", content: "Write and run Hugging Face transformers pipelines live in your browser." },
      { property: "og:title", content: "Transformers Playground" },
      { property: "og:description", content: "Write and run transformers pipelines live in your browser." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Playground,
});

type Line = { kind: "out" | "err" | "info"; text: string };

const fmt = (v: unknown) => {
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v, (_k, x) => (typeof x === "bigint" ? x.toString() : x), 2);
  } catch {
    return String(v);
  }
};

function Playground() {
  const [active, setActive] = useState(EXAMPLES[0]!.id);
  const [code, setCode] = useState(EXAMPLES[0]!.code);
  const [lines, setLines] = useState<Line[]>([]);
  const [running, setRunning] = useState(false);
  const libRef = useRef<typeof import("@huggingface/transformers") | null>(null);

  const push = (l: Line) => setLines((p) => [...p, l]);

  async function run() {
    setRunning(true);
    setLines([{ kind: "info", text: "▶ Running…" }]);
    const t0 = performance.now();
    try {
      if (!libRef.current) {
        push({ kind: "info", text: "Loading transformers library…" });
        libRef.current = await import("@huggingface/transformers");
      }
      const lib = libRef.current;
      const seen = new Set<string>();
      const pipeline = (task: string, model?: string, opts: Record<string, unknown> = {}) =>
        lib.pipeline(task as never, model, {
          ...opts,
          progress_callback: (p: { status: string; file?: string }) => {
            const key = `${p.status}:${p.file ?? ""}`;
            if ((p.status === "initiate" || p.status === "done") && !seen.has(key)) {
              seen.add(key);
              push({ kind: "info", text: `${p.status === "initiate" ? "↓" : "✓"} ${p.file ?? ""}` });
            }
          },
        } as never);
      const log = (...args: unknown[]) => push({ kind: "out", text: args.map(fmt).join(" ") });
      const AsyncFn = Object.getPrototypeOf(async () => {}).constructor;
      await new AsyncFn("pipeline", "log", "transformers", code)(pipeline, log, lib);
      push({ kind: "info", text: `✔ Done in ${((performance.now() - t0) / 1000).toFixed(2)}s` });
    } catch (e) {
      push({ kind: "err", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-4">
        <div>
          <h1 className="font-display text-xl font-bold tracking-tight">
            <span className="text-primary">🤗</span> transformers<span className="text-primary">.playground</span>
          </h1>
          <p className="text-xs text-muted-foreground">Real models, running in your browser. First run downloads model files.</p>
        </div>
        <button onClick={run} disabled={running} className="btn-run">
          {running ? "Running…" : "Run ▶"}
        </button>
      </header>

      <nav className="flex gap-2 overflow-x-auto border-b border-border px-5 py-3">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.id}
            onClick={() => { setActive(ex.id); setCode(ex.code); setLines([]); }}
            className={active === ex.id ? "chip chip-active" : "chip"}
          >
            {ex.label}
          </button>
        ))}
      </nav>

      <main className="grid flex-1 grid-cols-1 lg:grid-cols-2">
        <section className="flex flex-col border-b border-border lg:border-b-0 lg:border-r">
          <div className="panel-label">editor.js</div>
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
              if (e.key === "Tab") {
                e.preventDefault();
                const t = e.currentTarget, s = t.selectionStart;
                setCode(code.slice(0, s) + "  " + code.slice(t.selectionEnd));
                requestAnimationFrame(() => (t.selectionStart = t.selectionEnd = s + 2));
              }
            }}
            spellCheck={false}
            className="code-area min-h-[45vh] flex-1"
          />
        </section>
        <section className="flex flex-col">
          <div className="panel-label">output</div>
          <div className="code-area min-h-[35vh] flex-1 overflow-auto">
            {lines.length === 0 && <p className="text-muted-foreground">Press Run (or ⌘/Ctrl + Enter) to execute.</p>}
            {lines.map((l, i) => (
              <pre key={i} className={l.kind === "err" ? "text-destructive" : l.kind === "info" ? "text-muted-foreground" : "text-foreground"}>
                {l.text}
              </pre>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
