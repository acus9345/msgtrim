import { test } from "node:test";
import assert from "node:assert/strict";
import { trimChat, estimateTokens } from "../dist/index.js";

const one = { countTokens: () => 1 };
const sys = (content) => ({ role: "system", content });
const user = (content) => ({ role: "user", content });
const asst = (content) => ({ role: "assistant", content });

// OpenAI Chat Completions tool round trip
const oaiCall = (...ids) => ({
  role: "assistant",
  content: null,
  tool_calls: ids.map((id) => ({ id, type: "function", function: { name: "f", arguments: "{}" } })),
});
const oaiResult = (id) => ({ role: "tool", tool_call_id: id, content: "ok" });

// Anthropic Messages tool round trip
const antCall = (id) => ({ role: "assistant", content: [{ type: "tool_use", id, name: "f", input: {} }] });
const antResult = (id) => ({ role: "user", content: [{ type: "tool_result", tool_use_id: id, content: "ok" }] });

test("returns everything when it already fits", () => {
  const messages = [sys("s"), user("a"), asst("b")];
  const out = trimChat(messages, 100, one);
  assert.deepEqual(out, messages);
  assert.notEqual(out, messages);
});

test("keeps system messages and drops the oldest turns", () => {
  const messages = [sys("s"), user("1"), asst("1"), user("2"), asst("2"), user("3"), asst("3")];
  assert.deepEqual(trimChat(messages, 5, one), [sys("s"), user("2"), asst("2"), user("3"), asst("3")]);
  assert.deepEqual(trimChat(messages, 4, one), [sys("s"), user("3"), asst("3")]);
});

test("drops whole turns so the result starts with a user message", () => {
  const messages = [user("1"), asst("1"), user("2"), asst("2a"), asst("2b")];
  // Turn 2 costs 3, turn 1 costs 2; a budget of 4 cannot take turn 1 whole.
  assert.deepEqual(trimChat(messages, 4, one), [user("2"), asst("2a"), asst("2b")]);
});

test("never separates an OpenAI tool call from its results", () => {
  const messages = [
    user("1"),
    oaiCall("a", "b"),
    oaiResult("a"),
    oaiResult("b"),
    asst("1"),
    user("2"),
    asst("2"),
  ];
  assert.deepEqual(trimChat(messages, 6, one), [user("2"), asst("2")]);
  assert.deepEqual(trimChat(messages, 7, one), messages);
});

test("drops old tool round trips inside a long agent turn", () => {
  const messages = [
    sys("s"),
    user("go"),
    antCall("1"),
    antResult("1"),
    antCall("2"),
    antResult("2"),
    antCall("3"),
    antResult("3"),
    asst("done"),
  ];
  assert.deepEqual(trimChat(messages, 5, one), [sys("s"), user("go"), antCall("3"), antResult("3"), asst("done")]);
});

test("keeps a pending tool call at the end", () => {
  const messages = [user("go"), oaiCall("1"), oaiResult("1"), oaiCall("2"), oaiResult("2")];
  assert.deepEqual(trimChat(messages, 3, one), [user("go"), oaiCall("2"), oaiResult("2")]);
});

test("throws a RangeError when the minimum does not fit", () => {
  const messages = [sys("s"), user("go"), asst("done")];
  assert.throws(() => trimChat(messages, 2, one), RangeError);
  assert.throws(() => trimChat([sys("s"), sys("t")], 1, one), RangeError);
});

test("handles empty and system-only input", () => {
  assert.deepEqual(trimChat([], 0), []);
  assert.deepEqual(trimChat([sys("s")], 10, one), [sys("s")]);
});

test("does not modify the input", () => {
  const messages = [user("1"), asst("1"), user("2"), asst("2")];
  const copy = structuredClone(messages);
  const out = trimChat(messages, 2, one);
  assert.deepEqual(messages, copy);
  assert.equal(out[0], messages[2]);
});

test("uses the default estimator", () => {
  const messages = [user("x".repeat(400)), asst("y".repeat(400)), user("hi")];
  assert.deepEqual(trimChat(messages, 120), [user("hi")]);
});

test("rejects bad arguments with helpful messages", () => {
  assert.throws(() => trimChat("nope", 10), /must be an array/);
  assert.throws(() => trimChat([], -1), /non-negative number/);
  assert.throws(() => trimChat([{ content: "x" }], 10), /string "role"/);
  assert.throws(() => trimChat([user("x")], 10, { countTokens: () => NaN }), /countTokens returned NaN/);
});

test("estimateTokens counts text and gives media a flat cost", () => {
  assert.equal(estimateTokens(user("")), 5);
  assert.equal(estimateTokens(user("x".repeat(400))), 105);
  const image = { type: "image", source: { type: "base64", media_type: "image/png", data: "A".repeat(100000) } };
  assert.ok(estimateTokens(user([image])) < 1600);
});

test("random conversations always come back valid and within budget", () => {
  let seed = 42;
  const rand = (n) => {
    seed = (seed * 48271) % 2147483647;
    return seed % n;
  };
  for (let run = 0; run < 500; run++) {
    const anthropic = run % 2 === 0;
    const messages = rand(2) ? [sys("s")] : [];
    let id = 0;
    for (let turn = 0, turns = 1 + rand(5); turn < turns; turn++) {
      messages.push(user(`u${turn}`));
      for (let k = rand(4); k > 0; k--) {
        const callId = `t${id++}`;
        if (anthropic) messages.push(antCall(callId), antResult(callId));
        else messages.push(oaiCall(callId), oaiResult(callId));
      }
      if (rand(4)) messages.push(asst(`a${turn}`));
    }
    const budget = rand(messages.length + 2);
    let out;
    try {
      out = trimChat(messages, budget, one);
    } catch (err) {
      assert.ok(err instanceof RangeError);
      continue;
    }
    assert.ok(out.length <= budget);
    assert.equal(out.at(-1), messages.at(-1));
    const rest = out.filter((m) => m.role !== "system");
    assert.equal(rest[0].role, "user");
    assert.ok(!Array.isArray(rest[0].content), "first message is a plain user message");
    const calls = new Set();
    const results = new Set();
    for (const m of rest) {
      for (const c of m.tool_calls ?? []) calls.add(c.id);
      if (m.role === "tool") results.add(m.tool_call_id);
      for (const b of Array.isArray(m.content) ? m.content : []) {
        if (b.type === "tool_use") calls.add(b.id);
        if (b.type === "tool_result") results.add(b.tool_use_id);
      }
    }
    assert.deepEqual([...calls].sort(), [...results].sort());
  }
});
