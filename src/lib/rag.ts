// Мини-RAG на клиенте: чанкинг -> TF-IDF индекс -> косинусное сходство -> top-k.
// Никаких внешних API: всё считается прямо в браузере.

export interface Chunk {
  id: number;
  start: number; // позиция в исходном тексте (включительно)
  end: number; // позиция в исходном тексте (не включительно)
  text: string;
}

export interface ScoredChunk extends Chunk {
  score: number;
  matched: string[]; // основы слов запроса, найденные в чанке
  rank: number; // 1..N среди чанков с score > 0, 0 — если совпадений нет
}

export interface SearchOutcome {
  chunks: Chunk[];
  vocabSize: number;
  queryTerms: string[]; // основы (стеммы) слов запроса
  queryTermInfo: { term: string; df: number; idf: number }[];
  ranked: ScoredChunk[]; // только чанки с score > 0, по убыванию
  top: ScoredChunk[]; // первые k из ranked
}

const STOP = new Set(
  (
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это чтo " +
    "the a an and or of to in on for is are was were be been it its this that with as by at from what how why which who whom do does did can could should would will not no"
  ).split(/\s+/)
);

const RU_ENDINGS = [
  "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими", "ать", "ять", "ить", "еть", "ует", "уют", "ает", "яет", "ают", "яют",
  "ах", "ях", "ов", "ев", "ей", "ой", "ом", "ем", "ам", "ям", "ую", "юю", "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ых", "их", "ия", "ие", "ии", "ть",
  "а", "я", "о", "е", "ы", "и", "у", "ю", "ь", "й",
];
const EN_ENDINGS = ["ing", "ed", "es", "s"];

/** Лёгкий стеммер (отрезает типичные окончания). Достаточно для демонстрации, не претендует на полноту. */
export function stem(word: string): string {
  if (/^[a-z]+$/.test(word)) {
    for (const e of EN_ENDINGS) {
      if (word.length - e.length >= 3 && word.endsWith(e)) return word.slice(0, -e.length);
    }
    return word;
  }
  if (/^[а-я]+$/.test(word)) {
    for (const e of RU_ENDINGS) {
      if (word.length - e.length >= 3 && word.endsWith(e)) return word.slice(0, -e.length);
    }
  }
  return word;
}

export const WORD_RE = /[\p{L}\p{N}]+/gu;

export function normalizeWord(w: string): string {
  return w.toLowerCase().replace(/ё/g, "е");
}

/** Токены → основы, без стоп-слов. */
export function tokenize(text: string): string[] {
  const out: string[] = [];
  for (const m of text.matchAll(WORD_RE)) {
    const w = normalizeWord(m[0]);
    if (STOP.has(w)) continue;
    if (w.length < 2 && !/\d/.test(w)) continue;
    out.push(stem(w));
  }
  return out;
}

/**
 * Разбиение на чанки по словам: чанк не длиннее `size` символов, слова не рвутся
 * (кроме слов, которые сами длиннее size). overlap — сколько символов в конце чанка
 * повторяется в начале следующего.
 */
export function chunkText(text: string, size: number, overlap = 0): Chunk[] {
  if (!text.trim()) return [];
  const units: { s: number; e: number }[] = [];
  for (const m of text.matchAll(/\S+/g)) {
    let s = m.index ?? 0;
    const e = s + m[0].length;
    let cur = s;
    while (e - cur > size) {
      units.push({ s: cur, e: cur + size });
      cur += size;
    }
    units.push({ s: cur, e });
    s = e;
  }
  const chunks: Chunk[] = [];
  let i = 0;
  while (i < units.length) {
    const start = units[i].s;
    let j = i;
    while (j + 1 < units.length && units[j + 1].e - start <= size) j++;
    const end = units[j].e;
    chunks.push({ id: chunks.length, start, end, text: text.slice(start, end) });
    if (j >= units.length - 1) break;
    let k = j + 1;
    if (overlap > 0) {
      while (k - 1 > i && end - units[k - 1].s <= overlap) k--;
    }
    i = k;
  }
  return chunks;
}

/** Полный пайплайн: индексация + поиск. Вес = (1+ln tf) * idf, idf = ln((N+1)/(df+1)) + 1. */
export function search(text: string, query: string, size: number, overlap: number, k: number): SearchOutcome {
  const chunks = chunkText(text, size, overlap);
  const N = chunks.length;
  const docTokens = chunks.map((c) => tokenize(c.text));

  const df = new Map<string, number>();
  const tfs = docTokens.map((toks) => {
    const tf = new Map<string, number>();
    for (const t of toks) tf.set(t, (tf.get(t) ?? 0) + 1);
    for (const t of tf.keys()) df.set(t, (df.get(t) ?? 0) + 1);
    return tf;
  });
  const idf = (t: string) => Math.log((N + 1) / ((df.get(t) ?? 0) + 1)) + 1;

  const qTokens = tokenize(query);
  const qTf = new Map<string, number>();
  for (const t of qTokens) qTf.set(t, (qTf.get(t) ?? 0) + 1);
  const queryTerms = [...qTf.keys()];
  const qVec = new Map<string, number>();
  let qNorm = 0;
  for (const [t, f] of qTf) {
    const w = (1 + Math.log(f)) * idf(t);
    qVec.set(t, w);
    qNorm += w * w;
  }
  qNorm = Math.sqrt(qNorm);

  const queryTermInfo = queryTerms.map((term) => ({ term, df: df.get(term) ?? 0, idf: idf(term) }));

  const scored: ScoredChunk[] = [];
  if (N > 0 && qNorm > 0) {
    chunks.forEach((c, idx) => {
      const tf = tfs[idx];
      let dot = 0;
      let dNorm = 0;
      for (const [t, f] of tf) {
        const w = (1 + Math.log(f)) * idf(t);
        dNorm += w * w;
        const qw = qVec.get(t);
        if (qw !== undefined) dot += qw * w;
      }
      if (dot > 0) {
        const matched = queryTerms.filter((t) => tf.has(t));
        scored.push({ ...c, score: dot / (qNorm * Math.sqrt(dNorm)), matched, rank: 0 });
      }
    });
    scored.sort((a, b) => b.score - a.score || a.id - b.id);
    scored.forEach((s, i) => (s.rank = i + 1));
  }

  return {
    chunks,
    vocabSize: df.size,
    queryTerms,
    queryTermInfo,
    ranked: scored,
    top: scored.slice(0, k),
  };
}

export const SAMPLE_TEXT = `Retrieval-Augmented Generation (RAG) — это архитектура, в которой языковая модель отвечает не только по своим параметрам, но и по найденным внешним документам. Сначала система ищет релевантные фрагменты в базе знаний, затем добавляет их в промпт, и только после этого модель генерирует ответ.

Чанкинг — это разбиение длинного документа на небольшие фрагменты, называемые чанками. Если чанк слишком маленький, он теряет контекст: в нём может остаться одно предложение без пояснений. Если чанк слишком большой, в него попадает много лишнего, и оценка релевантности размывается.

Для поиска каждый чанк превращают в вектор. В нейросетевых системах это эмбеддинги, а в классическом подходе — TF-IDF. Вес слова растёт, если оно часто встречается в чанке, и падает, если оно есть почти в каждом чанке. Сходство запроса и чанка считают как косинус угла между векторами.

Параметр top-k определяет, сколько лучших чанков вернёт поиск. При маленьком k ответ получается точным, но можно потерять нужную деталь. При большом k растёт полнота, но в контекст модели попадает шум, а промпт становится длиннее и дороже.

Векторные базы данных, например FAISS, Qdrant или Chroma, хранят эмбеддинги и быстро находят ближайших соседей. Для небольших коллекций достаточно обычного перебора всех чанков.

Галлюцинации — это уверенные, но неверные ответы модели. RAG снижает риск галлюцинаций, потому что ответ опирается на реальные источники, которые можно показать пользователю и проверить. Качество ответа напрямую зависит от качества поиска: если нужный чанк не найден, модель не сможет на него опереться.`;

export const SAMPLE_QUERY = "Что такое размер чанка и параметр top-k?";
