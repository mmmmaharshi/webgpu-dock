import type { ProgressBar } from "./types";

{
  const el = (): HTMLElement | null => document.getElementById("log");
  const { log: origLog, warn: origWarn } = console;
  console.log = (...args: unknown[]) => {
    origLog(...args);
    const d = el();
    if (d) d.textContent += args.join(" ") + "\n";
  };
  console.warn = (...args: unknown[]) => {
    origWarn(...args);
    const d = el();
    if (d) d.innerHTML += `<span class="warn">${args.join(" ")}</span>\n`;
  };
}

export const bar: ProgressBar = (() => {
  const fill = () =>
    document.getElementById("bar-fill") as HTMLDivElement | null;
  const label = () =>
    document.getElementById("bar-label") as HTMLDivElement | null;
  const statusEl = () =>
    document.getElementById("status") as HTMLDivElement | null;
  return {
    setStatus(msg: string) {
      const s = statusEl();
      if (s) s.textContent = msg;
    },
    setProgress(pct: number) {
      const f = fill(),
        l = label();
      if (f) f.style.width = Math.min(pct, 100) + "%";
      if (l) l.textContent = Math.floor(Math.min(pct, 100)) + "%";
    },
    done() {
      this.setProgress(100);
      this.setStatus("Done");
    },
    async run<T>(
      initialStatus: string,
      fn: (pg: (pct: number) => void, st: (msg: string) => void) => Promise<T>,
    ): Promise<T> {
      this.setStatus(initialStatus);
      this.setProgress(0);
      const result = await fn(
        (pct: number) => this.setProgress(pct),
        (msg: string) => this.setStatus(msg),
      );
      return result;
    },
  };
})();
