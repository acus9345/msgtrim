# chattrim

Trim LLM chat history to a token budget while keeping it valid to send to the API.

Give it a message array and a token budget; it gives back a shorter array that still works with the API. It never orphans a tool result, never leaves a tool call without its result, and never starts the conversation with an assistant message.

Works with OpenAI Chat Completions messages (`tool_calls` + `role: "tool"`) and Anthropic Messages (`tool_use` + `tool_result` blocks). No dependencies.

## Install

```sh
npm install chattrim
```

## Quick start

```js
import { trimChat } from "chattrim";

const messages = [
  { role: "system", content: "You are terse." },
  { role: "user", content: "Tell me a long story." },
  { role: "assistant", content: "Once upon a time... ".repeat(500) },
  { role: "user", content: "Now summarise it." },
];

const trimmed = trimChat(messages, 200);
console.log(trimmed.map((m) => m.role)); // [ 'system', 'user' ]
```

## What gets kept

1. Leading `system` / `developer` messages are always kept.
2. The latest user message and everything after it are always kept.
3. Older history is added back newest-first, one whole turn at a time, while it fits.
4. A tool call and its results are kept or dropped together.
5. If the latest turn alone is over budget (a long agent loop), its oldest tool round trips are dropped, keeping the user message and the final message.

If even that minimum doesn't fit, `trimChat` throws a `RangeError` that says how many tokens it needs.

The input array and its messages are never modified; the result holds the same message objects.

## Accurate token counts

By default tokens are estimated at about 4 characters per token, plus a flat 1500 per image, audio clip or document. For exact counts, pass your own counter:

```js
import { trimChat } from "chattrim";
import { encode } from "gpt-tokenizer";

const trimmed = trimChat(messages, 8000, {
  countTokens: (m) => 4 + encode(typeof m.content === "string" ? m.content : JSON.stringify(m.content)).length,
});
```

Leave room in the budget for anything sent outside the messages: Anthropic's top-level `system` prompt, tool definitions, and the response (`max_tokens`).

## API

### `trimChat(messages, budget, options?)`

- `messages`: array of `{ role, content, ... }` objects.
- `budget`: maximum total tokens for the returned messages.
- `options.countTokens(message)`: token cost of one message. Defaults to `estimateTokens`.

Returns a new array. Throws `TypeError` for bad arguments and `RangeError` when the minimum doesn't fit.

### `estimateTokens(message)`

The default rough estimate for one message.

## Build from source

```sh
git clone https://github.com/acus9345/chattrim.git
cd chattrim
npm install
npm run build
npm test
```

## License

MIT
