/**
 * A chat message in OpenAI Chat Completions or Anthropic Messages shape.
 * Only `role`, `content` and `tool_calls` are looked at; everything else is passed through untouched.
 */
export interface ChatMessage {
  role: string;
  content?: unknown;
  [key: string]: unknown;
}

export interface TrimOptions<M> {
  /** Token cost of one message. Defaults to {@link estimateTokens}, a rough chars/4 heuristic. */
  countTokens?: (message: M) => number;
}

type Kind = "user" | "tool" | "other";

/** Messages that must be kept or dropped together: [start, end) in the input array. */
interface Unit {
  start: number;
  end: number;
  kind: Kind;
  tokens: number;
}

const MESSAGE_OVERHEAD = 4;
const MEDIA_TOKENS = 1500;
const MEDIA_TYPES = new Set(["image", "image_url", "input_image", "input_audio", "document", "file"]);

/**
 * Trim a conversation to fit a token budget while keeping it valid to send to the API.
 *
 * - Leading `system` / `developer` messages are always kept.
 * - The latest user message and everything after it are always kept.
 * - Older history is dropped oldest-first, one whole turn at a time, so the result
 *   still starts with a user message.
 * - A tool call is never separated from its results (OpenAI `tool_calls` + `tool`
 *   messages, Anthropic `tool_use` + `tool_result` blocks).
 * - If the latest turn alone is too big, its oldest tool round trips are dropped,
 *   keeping the user message and the final message.
 *
 * Returns a new array holding the same message objects; the input is not modified.
 * Throws a RangeError when even that minimum does not fit.
 *
 * @example trimChat(messages, 8000)
 * @example trimChat(messages, 8000, { countTokens: (m) => myTokenizer(m) })
 */
export function trimChat<M extends ChatMessage>(
  messages: readonly M[],
  budget: number,
  options: TrimOptions<M> = {},
): M[] {
  if (!Array.isArray(messages)) {
    throw new TypeError("trimChat(messages, budget): messages must be an array of { role, content } objects.");
  }
  if (typeof budget !== "number" || !Number.isFinite(budget) || budget < 0) {
    throw new TypeError(`trimChat(messages, budget): budget must be a non-negative number of tokens, got ${String(budget)}.`);
  }
  messages.forEach((m, i) => {
    if (m === null || typeof m !== "object" || typeof m.role !== "string") {
      throw new TypeError(`trimChat: message ${i} must be an object with a string "role".`);
    }
  });

  const count = options.countTokens ?? estimateTokens;
  const cost = messages.map((m, i) => {
    const n = count(m);
    if (typeof n !== "number" || !Number.isFinite(n) || n < 0) {
      throw new TypeError(`trimChat: countTokens returned ${String(n)} for message ${i}; it must return a non-negative number.`);
    }
    return n;
  });

  let pinnedEnd = 0;
  while (pinnedEnd < messages.length && isSystem(messages[pinnedEnd])) pinnedEnd++;
  const pinned = messages.slice(0, pinnedEnd);
  const units = toUnits(messages, pinnedEnd, cost);

  let anchor = units.length - 1;
  while (anchor > 0 && units[anchor].kind !== "user") anchor--;

  let used = sum(cost, 0, pinnedEnd) + units.slice(anchor).reduce((n, u) => n + u.tokens, 0);

  if (used <= budget) {
    // Add older turns, newest first, while they fit.
    let from = Math.max(anchor, 0);
    while (from > 0) {
      let prev = from - 1;
      while (prev > 0 && units[prev].kind !== "user") prev--;
      const turn = units.slice(prev, from).reduce((n, u) => n + u.tokens, 0);
      if (used + turn > budget) break;
      used += turn;
      from = prev;
    }
    return units.length === 0 ? pinned : [...pinned, ...messages.slice(units[from].start)];
  }

  // The latest turn alone is too big: drop its oldest tool round trips, never the last message.
  const kept = units.slice(anchor);
  const first = kept[0]?.kind === "user" ? 1 : 0;
  for (let i = first; i < kept.length - 1 && used > budget; ) {
    if (kept[i].kind === "tool") {
      used -= kept[i].tokens;
      kept.splice(i, 1);
    } else {
      i++;
    }
  }
  if (used > budget) {
    throw new RangeError(
      `trimChat: cannot fit the conversation in ${budget} tokens. The system messages, the latest user message ` +
        `and the final message alone need ${used}. Raise the budget or shorten those messages.`,
    );
  }
  return [...pinned, ...kept.flatMap((u) => messages.slice(u.start, u.end))];
}

/**
 * Rough token estimate for one message: about 4 characters per token plus a small
 * per-message overhead, and a flat cost for each image, audio clip or document.
 * Pass a real tokenizer through `countTokens` when you need accuracy.
 */
export function estimateTokens(message: unknown): number {
  return MESSAGE_OVERHEAD + Math.ceil(weigh(message));
}

function weigh(value: unknown): number {
  if (typeof value === "string") return value.length / 4;
  if (typeof value === "number" || typeof value === "boolean") return String(value).length / 4;
  if (Array.isArray(value)) return value.reduce((n: number, v) => n + weigh(v), 0);
  if (value !== null && typeof value === "object") {
    const type = (value as { type?: unknown }).type;
    if (typeof type === "string" && MEDIA_TYPES.has(type)) return MEDIA_TOKENS;
    return Object.values(value).reduce((n: number, v) => n + weigh(v), 0);
  }
  return 0;
}

function toUnits(messages: readonly ChatMessage[], from: number, cost: number[]): Unit[] {
  const units: Unit[] = [];
  let i = from;
  while (i < messages.length) {
    const start = i;
    const m = messages[i++];
    let kind: Kind;
    if (isToolCall(m)) {
      kind = "tool";
      while (i < messages.length && messages[i].role === "tool") i++;
      if (i < messages.length && isToolResult(messages[i])) i++;
    } else if (m.role === "tool" || isToolResult(m)) {
      kind = "tool";
    } else {
      kind = m.role === "user" ? "user" : "other";
    }
    units.push({ start, end: i, kind, tokens: sum(cost, start, i) });
  }
  return units;
}

function isSystem(m: ChatMessage): boolean {
  return m.role === "system" || m.role === "developer";
}

function isToolCall(m: ChatMessage): boolean {
  if (m.role !== "assistant") return false;
  return (Array.isArray(m.tool_calls) && m.tool_calls.length > 0) || hasBlock(m.content, "tool_use");
}

function isToolResult(m: ChatMessage): boolean {
  return m.role === "user" && hasBlock(m.content, "tool_result");
}

function hasBlock(content: unknown, type: string): boolean {
  return Array.isArray(content) && content.some((b) => b !== null && typeof b === "object" && b.type === type);
}

function sum(values: number[], start: number, end: number): number {
  let n = 0;
  for (let i = start; i < end; i++) n += values[i];
  return n;
}
