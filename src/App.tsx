import { useMemo, useRef, useState, type ReactNode } from "react";
import {
  SAMPLE_QUERY,
  SAMPLE_TEXT,
  WORD_RE,
  normalizeWord,
  search,
  stem,
  type ScoredChunk,
} from "./lib/rag";

const MIN_CHUNK = 10;
const MAX_CHUNK = 200;
const MIN_K = 1;
const MAX_K = 10;
const MAX_FILE = 1_000_000; // 1 МБ
const MAP_LIMIT = 1500; // максимум чанков в карте документа

const PALETTE = [
  "160 70% 45%",
  "199 85% 55%",
  "262 80% 68%",
  "38 92% 55%",
  "330 80% 65%",
  "20 90% 58%",
  "180 65% 45%",
  "95 60% 50%",
  "290 65% 62%",
  "55 85% 55%",
];
const color = (rank: number, a = 1) => `hsl(${PALETTE[(rank - 1) % PALETTE.length]} / ${a})`;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/* ---------- мелкие компоненты ---------- */

function NumberField({
  value,
  min,
  max,
  onChange,
  id,
}: {
  value: number;
  min: number;
  max: number;
  onChange: (n: number) => void;
  id: string;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <input
      id={id}
      type="number"
      min={min}
      max={max}
      value={draft ?? String(value)}
      onChange={(e) => {
        setDraft(e.target.value);
        const n = Number(e.target.value);
        if (e.target.value !== "" && Number.isFinite(n) && n >= min && n <= max) onChange(Math.round(n));
      }}
      onBlur={() => {
        if (draft !== null) {
          const n = Number(draft);
          if (draft !== "" && Number.isFinite(n)) onChange(clamp(Math.round(n), min, max));
          setDraft(null);
        }
      }}
      className="w-20 rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-right font-mono text-sm text-emerald-300 outline-none focus:border-emerald-500"
    />
  );
}

function Card({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 shadow-lg shadow-black/20">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">{title}</h2>
        {right}
      </div>
      {children}
    </section>
  );
}

function Highlighted({ text, terms }: { text: string; terms: Set<string> }) {
  const nodes: ReactNode[] = [];
  let last = 0;
  let n = 0;
  for (const m of text.matchAll(WORD_RE)) {
    const idx = m.index ?? 0;
    if (idx > last) nodes.push(text.slice(last, idx));
    if (terms.has(stem(normalizeWord(m[0])))) {
      nodes.push(
        <mark key={n++} className="rag-hit">
          {m[0]}
        </mark>
      );
    } else nodes.push(m[0]);
    last = idx + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return <>{nodes}</>;
}

function Notice({ tone, children }: { tone: "info" | "warn" | "empty"; children: ReactNode }) {
  const cls =
    tone === "warn"
      ? "border-amber-500/40 bg-amber-500/10 text-amber-200"
      : tone === "empty"
        ? "border-slate-700 bg-slate-800/40 text-slate-300"
        : "border-sky-500/30 bg-sky-500/10 text-sky-200";
  return <div className={`rounded-xl border px-3 py-2 text-sm ${cls}`}>{children}</div>;
}

/* ---------- приложение ---------- */

export default function App() {
  const [text, setText] = useState(SAMPLE_TEXT);
  const [query, setQuery] = useState(SAMPLE_QUERY);
  const [size, setSize] = useState(50);
  const [k, setK] = useState(2);
  const [overlapRaw, setOverlapRaw] = useState(0);
  const [showRest, setShowRest] = useState(true);
  const [fileError, setFileError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const overlap = Math.min(overlapRaw, Math.floor(size / 2));

  const out = useMemo(() => search(text, query, size, overlap, k), [text, query, size, overlap, k]);

  // история результатов — чтобы показывать, какие фрагменты появились после смены параметров
  const hist = useRef<{ curr: typeof out; prev: typeof out | null }>({ curr: out, prev: null });
  if (hist.current.curr !== out) hist.current = { curr: out, prev: hist.current.curr };
  const prevTexts = hist.current.prev ? new Set(hist.current.prev.top.map((c) => c.text)) : null;
  const newCount = prevTexts ? out.top.filter((c) => !prevTexts.has(c.text)).length : 0;

  const termSet = useMemo(() => new Set(out.queryTerms), [out.queryTerms]);
  const topIds = useMemo(() => new Map(out.top.map((c) => [c.id, c])), [out.top]);
  const rankedMap = useMemo(() => new Map(out.ranked.map((c) => [c.id, c])), [out.ranked]);

  const textEmpty = text.trim().length === 0;
  const queryEmpty = query.trim().length === 0;
  const noTerms = !queryEmpty && out.queryTerms.length === 0;
  const textShort = !textEmpty && text.trim().length < size;
  const canSearch = !textEmpty && !queryEmpty && !noTerms;
  const rest = out.ranked.slice(k, k + 8);

  const avgLen = out.chunks.length ? Math.round(out.chunks.reduce((s, c) => s + c.text.length, 0) / out.chunks.length) : 0;
  const context = out.top.map((c) => c.text).join("\n---\n");

  function onFile(f: File | undefined) {
    setFileError(null);
    if (!f) return;
    if (f.size > MAX_FILE) {
      setFileError("Файл больше 1 МБ — возьмите файл поменьше.");
      return;
    }
    if (f.type && !f.type.startsWith("text/") && !/json|xml|markdown/.test(f.type)) {
      setFileError("Поддерживаются только текстовые файлы (.txt, .md, .csv, .json).");
      return;
    }
    const r = new FileReader();
    r.onload = () => {
      const s = String(r.result ?? "");
      if (s.includes("\u0000")) {
        setFileError("Похоже, это бинарный файл, а не текст.");
        return;
      }
      setText(s);
      setFileName(f.name);
    };
    r.onerror = () => setFileError("Не удалось прочитать файл.");
    r.readAsText(f, "utf-8");
  }

  /* карта документа */
  const docMap = useMemo(() => {
    if (out.chunks.length === 0 || out.chunks.length > MAP_LIMIT) return null;
    const parts: ReactNode[] = [];
    let covered = 0;
    out.chunks.forEach((c, i) => {
      if (c.start > covered) parts.push(<span key={`g${i}`}>{text.slice(covered, c.start)}</span>);
      const from = Math.max(c.start, covered);
      const shown = text.slice(from, c.end);
      covered = Math.max(covered, c.end);
      const top = topIds.get(c.id);
      const r = rankedMap.get(c.id);
      const style: React.CSSProperties = top
        ? { background: color(top.rank, 0.28), boxShadow: `inset 0 -2px 0 ${color(top.rank)}`, color: "#f8fafc" }
        : r
          ? { background: "rgba(148,163,184,0.12)", boxShadow: "inset 0 -2px 0 rgba(148,163,184,0.5)" }
          : { background: i % 2 ? "rgba(30,41,59,0.8)" : "rgba(15,23,42,0.8)" };
      parts.push(
        <span
          key={c.id}
          title={`Чанк #${c.id + 1}${top ? ` — место ${top.rank}, score ${top.score.toFixed(4)}` : r ? ` — score ${r.score.toFixed(4)} (не вошёл в top-k)` : " — нет совпадений"}`}
          style={style}
          className="rounded-sm border-l border-slate-600 px-0.5 transition-colors duration-300"
        >
          {top && (
            <sup className="mr-0.5 font-mono text-[10px] font-bold" style={{ color: color(top.rank) }}>
              {top.rank}
            </sup>
          )}
          {shown}
        </span>
      );
    });
    if (covered < text.length) parts.push(<span key="tail">{text.slice(covered)}</span>);
    return parts;
  }, [out.chunks, text, topIds, rankedMap]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <header className="border-b border-slate-800 bg-gradient-to-r from-emerald-950/60 via-slate-950 to-sky-950/50">
        <div className="mx-auto max-w-7xl px-4 py-6">
          <div className="flex flex-wrap items-center gap-3">
            <span className="rounded-lg bg-emerald-500/15 px-2 py-1 font-mono text-xs text-emerald-300">RAG Lab</span>
            <h1 className="text-xl font-bold text-white sm:text-2xl">Как chunk size и top-k меняют найденные фрагменты</h1>
          </div>
          <p className="mt-2 max-w-3xl text-sm text-slate-400">
            Интерактивный тренажёр retrieval-части RAG. Документ режется на чанки, строится TF-IDF-индекс, запрос сравнивается
            с каждым чанком по косинусному сходству, и в «контекст для LLM» уходят k лучших. Всё считается в вашем браузере —
            без серверов и платных API. Двигайте ползунки — результат пересчитывается мгновенно.
          </p>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-4 px-4 py-6 lg:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
        {/* ---------- левая колонка: ввод ---------- */}
        <div className="space-y-4">
          <Card
            title="1 · Документ"
            right={
              <div className="flex gap-1.5">
                <button
                  onClick={() => {
                    setText(SAMPLE_TEXT);
                    setQuery(SAMPLE_QUERY);
                    setFileName(null);
                    setFileError(null);
                  }}
                  className="rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
                >
                  Пример
                </button>
                <button
                  onClick={() => fileRef.current?.click()}
                  className="rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
                >
                  Загрузить .txt
                </button>
                <button
                  onClick={() => {
                    setText("");
                    setFileName(null);
                  }}
                  className="rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
                >
                  Очистить
                </button>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".txt,.md,.csv,.json,text/*"
                  className="hidden"
                  onChange={(e) => {
                    onFile(e.target.files?.[0]);
                    e.target.value = "";
                  }}
                />
              </div>
            }
          >
            <textarea
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                setFileName(null);
              }}
              rows={9}
              placeholder="Вставьте сюда текст для индексации…"
              className="w-full resize-y rounded-xl border border-slate-700 bg-slate-950 p-3 text-sm leading-relaxed text-slate-200 outline-none placeholder:text-slate-600 focus:border-emerald-500"
            />
            <div className="mt-2 flex flex-wrap justify-between gap-2 text-xs text-slate-500">
              <span>
                {text.length.toLocaleString("ru")} симв.{fileName && <> · файл: <span className="text-slate-300">{fileName}</span></>}
              </span>
              <span>чанков: {out.chunks.length.toLocaleString("ru")}</span>
            </div>
            {fileError && <div className="mt-2"><Notice tone="warn">{fileError}</Notice></div>}
          </Card>

          <Card title="2 · Параметры чанкинга и поиска">
            <div className="space-y-5">
              <div>
                <div className="mb-1 flex items-center justify-between">
                  <label htmlFor="size" className="text-sm text-slate-300">
                    Размер чанка <span className="text-slate-500">(символов)</span>
                  </label>
                  <NumberField id="size" value={size} min={MIN_CHUNK} max={MAX_CHUNK} onChange={setSize} />
                </div>
                <input
                  type="range"
                  min={MIN_CHUNK}
                  max={MAX_CHUNK}
                  value={size}
                  onChange={(e) => setSize(Number(e.target.value))}
                  className="w-full"
                  aria-label="Размер чанка"
                />
                <div className="flex justify-between text-[10px] text-slate-600">
                  <span>{MIN_CHUNK}</span>
                  <span>{MAX_CHUNK}</span>
                </div>
              </div>

              <div>
                <div className="mb-1 flex items-center justify-between">
                  <label htmlFor="k" className="text-sm text-slate-300">
                    top-k <span className="text-slate-500">(сколько чанков вернуть)</span>
                  </label>
                  <NumberField id="k" value={k} min={MIN_K} max={MAX_K} onChange={setK} />
                </div>
                <input
                  type="range"
                  min={MIN_K}
                  max={MAX_K}
                  value={k}
                  onChange={(e) => setK(Number(e.target.value))}
                  className="w-full"
                  aria-label="top-k"
                />
                <div className="flex justify-between text-[10px] text-slate-600">
                  <span>{MIN_K}</span>
                  <span>{MAX_K}</span>
                </div>
              </div>

              <div>
                <div className="mb-1 flex items-center justify-between">
                  <label htmlFor="ov" className="text-sm text-slate-300">
                    Перекрытие <span className="text-slate-500">(overlap, необязательно)</span>
                  </label>
                  <NumberField id="ov" value={overlap} min={0} max={Math.floor(size / 2)} onChange={setOverlapRaw} />
                </div>
                <input
                  type="range"
                  min={0}
                  max={Math.floor(size / 2)}
                  value={overlap}
                  onChange={(e) => setOverlapRaw(Number(e.target.value))}
                  className="w-full"
                  aria-label="Перекрытие чанков"
                />
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                <p className="mb-2 text-xs text-slate-400">Сценарий проверки — переключайтесь и сравнивайте выдачу:</p>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => {
                      setSize(50);
                      setK(2);
                    }}
                    className="rounded-lg bg-emerald-500/15 px-3 py-1.5 text-xs font-medium text-emerald-300 hover:bg-emerald-500/25"
                  >
                    chunk 50 · k = 2
                  </button>
                  <button
                    onClick={() => {
                      setSize(150);
                      setK(5);
                    }}
                    className="rounded-lg bg-sky-500/15 px-3 py-1.5 text-xs font-medium text-sky-300 hover:bg-sky-500/25"
                  >
                    chunk 150 · k = 5
                  </button>
                </div>
              </div>
            </div>
          </Card>

          <Card title="3 · Поисковый запрос">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Например: что такое top-k?"
              className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-emerald-500"
            />
            <div className="mt-3 flex flex-wrap items-center gap-1.5 text-xs">
              <span className="text-slate-500">Основы слов запроса:</span>
              {out.queryTermInfo.length === 0 && <span className="text-slate-600">—</span>}
              {out.queryTermInfo.map((t) => (
                <span
                  key={t.term}
                  title={`df = ${t.df} из ${out.chunks.length} чанков, idf = ${t.idf.toFixed(3)}`}
                  className={`rounded-md px-1.5 py-0.5 font-mono ${t.df > 0 ? "bg-yellow-400/15 text-yellow-200" : "bg-slate-800 text-slate-500 line-through"}`}
                >
                  {t.term}
                  <span className="ml-1 text-[10px] opacity-70">df={t.df}</span>
                </span>
              ))}
            </div>
          </Card>
        </div>

        {/* ---------- правая колонка: результаты ---------- */}
        <div className="space-y-4">
          {/* пайплайн */}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {[
              { n: "Чанкинг", v: `${out.chunks.length}`, s: out.chunks.length ? `ср. длина ${avgLen} симв.` : "нет чанков" },
              { n: "Индекс TF-IDF", v: `${out.vocabSize}`, s: "уникальных терминов" },
              { n: "Совпадения", v: `${out.ranked.length}`, s: "чанков со score > 0" },
              { n: "Top-k", v: `${out.top.length}/${k}`, s: "вернулось / запрошено" },
            ].map((s, i) => (
              <div key={s.n} className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
                <div className="text-[11px] uppercase tracking-wider text-slate-500">
                  {i + 1}. {s.n}
                </div>
                <div className="font-mono text-2xl font-bold text-emerald-300">{s.v}</div>
                <div className="text-xs text-slate-500">{s.s}</div>
              </div>
            ))}
          </div>

          <Card
            title="Найденные фрагменты"
            right={
              <span className="rounded-md bg-slate-800 px-2 py-0.5 font-mono text-xs text-slate-400">
                chunk={size} · k={k}
                {overlap > 0 && ` · overlap=${overlap}`}
              </span>
            }
          >
            <div className="space-y-3">
              {textEmpty && <Notice tone="empty">📄 Документ пуст. Вставьте текст или загрузите файл — тогда появятся чанки для поиска.</Notice>}
              {!textEmpty && queryEmpty && <Notice tone="empty">🔎 Запрос пуст. Введите вопрос, чтобы найти релевантные фрагменты.</Notice>}
              {noTerms && (
                <Notice tone="warn">
                  В запросе нет значимых слов (только стоп-слова, знаки или одиночные буквы). Переформулируйте запрос.
                </Notice>
              )}
              {textShort && (
                <Notice tone="info">
                  Текст ({text.trim().length} симв.) короче размера чанка ({size}) — весь документ превращается в один чанк, поэтому
                  top-k ничего не меняет.
                </Notice>
              )}
              {canSearch && out.ranked.length === 0 && (
                <Notice tone="empty">
                  🙈 Совпадений нет: ни одно слово запроса не встречается ни в одном чанке. Результат пуст — это нормальное
                  поведение поиска, а не ошибка.
                </Notice>
              )}
              {canSearch && out.ranked.length > 0 && out.ranked.length < k && (
                <Notice tone="info">
                  Запрошено k = {k}, но релевантных чанков только {out.ranked.length}. Чанки с score = 0 не возвращаются.
                </Notice>
              )}
              {canSearch && out.top.length > 0 && prevTexts && newCount > 0 && (
                <Notice tone="info">
                  ✨ После изменения параметров появилось новых фрагментов: <b>{newCount}</b> (отмечены «новый»).
                </Notice>
              )}

              {out.top.map((c) => (
                <ResultItem key={`${size}-${overlap}-${c.id}`} c={c} terms={termSet} isNew={!!prevTexts && !prevTexts.has(c.text)} />
              ))}

              {canSearch && rest.length > 0 && (
                <div>
                  <button onClick={() => setShowRest((v) => !v)} className="text-xs text-slate-400 underline-offset-2 hover:underline">
                    {showRest ? "Скрыть" : "Показать"} чанки, которые не прошли отсечение top-k ({Math.min(8, out.ranked.length - k)})
                  </button>
                  {showRest && (
                    <div className="mt-2 space-y-2">
                      {rest.map((c) => (
                        <div key={`${size}-${c.id}`} className="rag-pop rounded-xl border border-dashed border-slate-700 bg-slate-950/40 p-3 opacity-60">
                          <div className="mb-1 flex flex-wrap items-center gap-2 text-xs">
                            <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-slate-400">#{c.rank}</span>
                            <span className="font-mono text-slate-400">score {c.score.toFixed(4)}</span>
                            <span className="text-slate-600">— отсечён: место {c.rank} &gt; k = {k}</span>
                          </div>
                          <p className="text-sm text-slate-400">{c.text}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </Card>

          {out.top.length > 0 && (
            <Card
              title="Контекст, который получит LLM"
              right={<span className="text-xs text-slate-500">{context.length} симв. ≈ {Math.ceil(context.length / 4)} токенов</span>}
            >
              <pre className="whitespace-pre-wrap rounded-xl bg-slate-950 p-3 font-mono text-xs leading-relaxed text-slate-300">
                {`Ответь на вопрос, используя только контекст.\n\nКонтекст:\n${context}\n\nВопрос: ${query.trim()}`}
              </pre>
            </Card>
          )}

          <Card
            title="Карта документа"
            right={<span className="text-xs text-slate-500">цвет = место в top-k, границы = чанки</span>}
          >
            {textEmpty ? (
              <p className="text-sm text-slate-500">Нет текста для отображения.</p>
            ) : docMap ? (
              <p className="max-h-72 overflow-auto whitespace-pre-wrap rounded-xl bg-slate-950 p-3 text-sm leading-8 text-slate-400">{docMap}</p>
            ) : (
              <p className="text-sm text-slate-500">
                Чанков слишком много ({out.chunks.length.toLocaleString("ru")}) для отрисовки карты — увеличьте размер чанка (лимит карты: {MAP_LIMIT}).
              </p>
            )}
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-500">
              <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: color(1, 0.6) }} /> в top-k</span>
              <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-slate-500/50" /> score &gt; 0, но не прошёл</span>
              <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-slate-800" /> нет совпадений</span>
            </div>
          </Card>

          <Card title="Как это посчитано (честная математика)">
            <div className="space-y-2 text-sm leading-relaxed text-slate-400">
              <p>
                <b className="text-slate-300">Это не нейросетевые эмбеддинги, а классический TF-IDF.</b> Принцип retrieval тот же, что у
                векторных баз: текст → вектор → ближайшие по косинусу.
              </p>
              <ol className="list-decimal space-y-1 pl-5">
                <li>Слова приводятся к нижнему регистру, убираются стоп-слова, окончания отсекаются лёгким стеммером.</li>
                <li>
                  Вес термина в чанке: <code className="rounded bg-slate-800 px-1 text-emerald-300">w = (1 + ln tf) · idf</code>, где{" "}
                  <code className="rounded bg-slate-800 px-1 text-emerald-300">idf = ln((N+1)/(df+1)) + 1</code>, N — число чанков, df — в скольких чанках встретился термин.
                </li>
                <li>
                  Score = <code className="rounded bg-slate-800 px-1 text-emerald-300">cos(q, d) = (q·d) / (‖q‖·‖d‖)</code> — число от 0 до 1.
                </li>
                <li>Чанки сортируются по score; возвращаются первые k с score &gt; 0.</li>
              </ol>
              <p>
                Почему меняется выдача: при коротком чанке слово из запроса «занимает» большую долю вектора, но контекста мало; при длинном —
                в чанк попадает несколько тем, ‖d‖ растёт и score падает. Число чанков N меняет idf, а k лишь определяет, где
                проходит отсечка.
              </p>
            </div>
          </Card>
        </div>
      </main>

      <footer className="border-t border-slate-800 py-5 text-center text-xs text-slate-600">
        RAG Lab · автономное приложение, все вычисления выполняются локально в браузере
      </footer>
    </div>
  );
}

function ResultItem({ c, terms, isNew }: { c: ScoredChunk; terms: Set<string>; isNew: boolean }) {
  return (
    <article
      className="rag-pop rounded-xl border bg-slate-950/60 p-3"
      style={{ borderColor: color(c.rank, 0.6), boxShadow: `0 0 0 1px ${color(c.rank, 0.12)}` }}
    >
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded-md px-2 py-0.5 font-mono text-sm font-bold text-slate-950" style={{ background: color(c.rank) }}>
          #{c.rank}
        </span>
        <span className="font-mono text-slate-200">
          score <b className="text-base text-white">{c.score.toFixed(4)}</b>
        </span>
        {isNew && <span className="rounded-full bg-yellow-400/20 px-2 py-0.5 font-medium text-yellow-200">новый</span>}
        <span className="ml-auto text-slate-500">
          чанк #{c.id + 1} · символы {c.start}–{c.end} · {c.text.length} симв.
        </span>
      </div>
      <div className="mb-2 h-1.5 overflow-hidden rounded-full bg-slate-800">
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.max(2, c.score * 100)}%`, background: color(c.rank) }} />
      </div>
      <p className="text-sm leading-relaxed text-slate-200">
        <Highlighted text={c.text} terms={terms} />
      </p>
      <div className="mt-2 flex flex-wrap gap-1 text-[11px] text-slate-500">
        совпавшие термины:
        {c.matched.map((m) => (
          <span key={m} className="rounded bg-yellow-400/10 px-1 font-mono text-yellow-200">
            {m}
          </span>
        ))}
      </div>
    </article>
  );
}
