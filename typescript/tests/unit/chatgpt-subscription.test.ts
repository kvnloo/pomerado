import { writeFileSync } from "node:fs";
import { mkdtemp, readFile, stat, writeFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, Runner, applyPatchTool } from "@openai/agents";
import patchFixture from "./fixtures/chatgpt-apply-patch.json" with { type: "json" };
import featuresFixture from "./fixtures/chatgpt-request-features.json" with { type: "json" };
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  ChatgptSubscriptionAuthError,
  accessTokenExpiring,
  chatgptSubscriptionProvider,
  modelProvidersFromEnv,
  redactSecrets,
  toCodexRequest,
} from "../../src/models/chatgpt-subscription.js";
import { nativePatchItem } from "../../src/models/chatgpt-patch.js";
import { prewarmInstructions } from "../../src/models/cached-instructions.js";
import { providerQuotaExhausted } from "../../src/models/provider-quota.js";

const jwt = (claims: Record<string, unknown>) =>
  `eyJhbGciOiJub25lIn0.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.c2lnbmF0dXJl`;
const nowSeconds = () => Math.floor(Date.now() / 1000);
const freshToken = (tag: string) =>
  jwt({ exp: nowSeconds() + 3600, tag, "https://api.openai.com/auth": { chatgpt_account_id: "acct-1" } });
const staleToken = (tag: string) => jwt({ exp: nowSeconds() - 10, tag });

interface Recorded {
  readonly path: string;
  readonly headers: IncomingMessage["headers"];
  readonly body: Record<string, unknown>;
}

const completed = (text: string) => ({
  id: "resp_1",
  object: "response",
  created_at: 1,
  status: "completed",
  model: "test-model",
  output: [
    {
      type: "message",
      id: "msg_1",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text, annotations: [] }],
    },
  ],
  usage: {
    input_tokens: 5,
    input_tokens_details: { cached_tokens: 0 },
    output_tokens: 2,
    output_tokens_details: { reasoning_tokens: 0 },
    total_tokens: 7,
  },
});
const sse = (...events: readonly Record<string, unknown>[]) =>
  events.map((event) => `event: ${String(event["type"])}\ndata: ${JSON.stringify(event)}\n\n`).join("");

type Reply = (request: Recorded) => { status: number; body: string; type?: string };

let server: Server;
let base: string;
let recorded: Recorded[];
let replies: Reply[];
let dir: string;

beforeEach(async () => {
  recorded = [];
  replies = [];
  dir = await mkdtemp(join(tmpdir(), "pomerado-chatgpt-"));
  server = createServer((request, response) => {
    let text = "";
    request.on("data", (chunk: Buffer) => (text += chunk.toString("utf8")));
    request.on("end", () => {
      const entry: Recorded = {
        path: request.url ?? "",
        headers: request.headers,
        body: text === "" ? {} : (JSON.parse(text) as Record<string, unknown>),
      };
      recorded.push(entry);
      const reply: Reply = replies.shift() ?? (() => ({ status: 500, body: "{}" }));
      const { status, body, type } = reply(entry);
      response.writeHead(status, { "content-type": type ?? "text/event-stream" });
      response.end(body);
    });
  });
  await new Promise<void>((resume) => server.listen(0, "127.0.0.1", resume));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
afterEach(async () => {
  await new Promise((resume) => server.close(resume));
});

const writeAuth = async (accessToken: string, refreshToken = "refresh-token-original") => {
  const path = join(dir, "auth.json");
  await writeFile(
    path,
    JSON.stringify({
      auth_mode: "chatgpt",
      OPENAI_API_KEY: null,
      tokens: { access_token: accessToken, account_id: "acct-1", id_token: "x", refresh_token: refreshToken },
    }),
  );
  return path;
};
const okText: Reply = () => ({
  status: 200,
  body: sse(
    { type: "response.created", response: { id: "resp_1" } },
    { type: "response.output_text.delta", delta: "pong" },
    { type: "response.completed", response: completed("pong") },
  ),
});

const run = async (provider: ReturnType<typeof chatgptSubscriptionProvider>, instructions = "Answer briefly.") => {
  const runner = new Runner({ modelProvider: provider, tracingDisabled: true });
  const agent = new Agent({ name: "probe", instructions, model: "test-model" });
  return runner.run(agent, "ping", { maxTurns: 1 });
};

describe("chatgpt subscription transport", () => {
  it("allow-lists request features and replays opaque compaction checkpoints without stored IDs", () => {
    expect(toCodexRequest(featuresFixture.request)).toEqual(featuresFixture.expected);
  });

  it.each(["compaction_trigger", "item_reference"])("rejects unsupported %s locally with typed error metadata", async (type) => {
    const authPath = await writeAuth(freshToken("unsupported"));
    const provider = chatgptSubscriptionProvider({ authPath, baseUrl: base });
    const error = await new Runner({ modelProvider: provider, tracingDisabled: true }).run(
      new Agent({ name: "unsupported probe", model: "test-model" }),
      [{ type: "unknown", providerData: { type, id: "unavailable_item" } }],
      { maxTurns: 1 },
    ).catch((error: unknown) => error);
    expect(error).toMatchObject({ status: 400, code: "unsupported_input_item", name: "Error" });
    expect(recorded).toHaveLength(0);
  });

  it("keeps HTTP rejection status, code and SDK error name", async () => {
    const authPath = await writeAuth(freshToken("rejected"));
    replies.push(() => ({ status: 400, body: JSON.stringify({ error: {
      message: patchFixture.rejection.detail, code: "unsupported_tool_type", type: "invalid_request_error",
    } }), type: "application/json" }));
    const error = await run(chatgptSubscriptionProvider({ authPath, baseUrl: base })).catch((error: unknown) => error);
    expect(error).toMatchObject({ status: 400, code: "unsupported_tool_type", name: "Error" });
    expect(recorded).toHaveLength(1);
  });
  it.each([false, true])("executes a native patch through the function-only backend and replays its result (stream=%s)", async (stream) => {
    const authPath = await writeAuth(freshToken("patch"));
    const executed: unknown[] = [];
    replies.push((request) => {
      const tools = request.body["tools"];
      if (Array.isArray(tools) && tools.some((tool) => tool.type === "apply_patch"))
        return { status: 400, body: JSON.stringify(patchFixture.rejection), type: "application/json" };
      expect(tools).toContainEqual(expect.objectContaining({ type: "function", name: "apply_patch" }));
      return {
        status: 200,
        body: sse(
          { type: "response.created", response: { id: "resp_1" } },
          { type: "response.output_item.added", output_index: 0, item: { ...patchFixture.call, arguments: "", status: "in_progress" } },
          { type: "response.function_call_arguments.delta", item_id: patchFixture.call.id, output_index: 0, delta: patchFixture.call.arguments },
          { type: "response.function_call_arguments.done", item_id: patchFixture.call.id, output_index: 0, arguments: patchFixture.call.arguments },
          { type: "response.output_item.done", output_index: 0, item: patchFixture.call },
          { type: "response.completed", response: { ...completed(""), output: [patchFixture.call] } },
        ),
      };
    });
    replies.push((request) => {
      expect(request.body["input"]).toContainEqual(patchFixture.functionOutput);
      expect(request.body["input"]).toContainEqual({
        ...patchFixture.call, id: undefined,
      });
      return okText(request);
    });
    const provider = chatgptSubscriptionProvider({ authPath, baseUrl: base });
    const runner = new Runner({ modelProvider: provider, tracingDisabled: true });
    const agent = new Agent({
      name: "patch probe", model: "test-model", instructions: "Create a file.",
      tools: [applyPatchTool({ editor: {
        createFile: async (operation) => { executed.push(operation); return { output: "created" }; },
        updateFile: async () => { throw new Error("Unexpected update"); },
        deleteFile: async () => { throw new Error("Unexpected delete"); },
      } })],
    });
    const result = stream
      ? await runner.run(agent, "Create probe.txt", { maxTurns: 2, stream: true })
      : await runner.run(agent, "Create probe.txt", { maxTurns: 2 });
    const patchEvents: unknown[] = [];
    if ("completed" in result) {
      for await (const event of result) {
        if (event.type === "raw_model_stream_event" && event.data.type === "model")
          patchEvents.push(event.data.event);
      }
      await result.completed;
      expect(patchEvents).toContainEqual(expect.objectContaining({
        type: "response.output_item.done", item: patchFixture.nativeCall,
      }));
      expect(patchEvents).not.toContainEqual(expect.objectContaining({ type: "response.function_call_arguments.delta" }));
    }
    expect(result.finalOutput).toBe("pong");
    expect(executed).toEqual([patchFixture.operation]);
    expect(result.rawResponses[0]?.output[0]).toMatchObject({
      type: "apply_patch_call", callId: "call_patch_1", operation: patchFixture.operation,
    });
    expect(recorded).toHaveLength(2);
  });

  it.each([
    { type: "create_file", path: "new.txt", diff: "+new\n" },
    { type: "update_file", path: "old.txt", diff: "@@\n-old\n+new\n" },
    { type: "update_file", path: "old.txt", diff: "@@\n-old\n+new\n", move_to: "new.txt" },
    { type: "delete_file", path: "old.txt" },
  ])("round trips the patch operation $type", (operation) => {
    const native = { ...patchFixture.nativeCall, operation };
    const body = toCodexRequest({
      input: [native, { ...patchFixture.nativeOutput, status: "failed", output: "not applied" }],
      tools: [patchFixture.tool], tool_choice: { type: "apply_patch" },
    });
    expect(body["tool_choice"]).toEqual({ type: "function", name: "apply_patch" });
    expect(body["input"]).toEqual([
      { type: "function_call", name: "apply_patch", call_id: "call_patch_1", status: "completed", arguments: JSON.stringify({ operation }) },
      { type: "function_call_output", call_id: "call_patch_1", output: JSON.stringify({ status: "failed", output: "not applied" }) },
    ]);
    expect(nativePatchItem({ ...patchFixture.call, arguments: JSON.stringify({ operation }) })).toEqual(native);
    if ("move_to" in operation)
      expect(JSON.stringify(toCodexRequest({ tools: [patchFixture.tool] })["tools"])).toContain("move_to");
  });

  it.each(["not json", "{}", '{"operation":{"type":"update_file","path":"x"}}'])("rejects malformed patch arguments: %s", (args) => {
    expect(() => nativePatchItem({ ...patchFixture.call, arguments: args })).toThrow(/apply_patch/);
  });

  it("keeps ordinary function calls unchanged", () => {
    const call = { ...patchFixture.call, name: "lookup" };
    expect(nativePatchItem(call)).toBe(call);
    expect(toCodexRequest({ input: [call], tools: [{ type: "function", name: "apply_patch", parameters: {} }] })["tools"])
      .toEqual([{ type: "function", name: "apply_patch", parameters: {} }]);
  });

  it("signs a non-streamed call, reshapes its body and returns the streamed final response", async () => {
    const token = freshToken("a");
    const authPath = await writeAuth(token);
    replies.push(okText);
    const result = await run(chatgptSubscriptionProvider({ authPath, baseUrl: base }));
    expect(result.finalOutput).toBe("pong");
    const [request] = recorded;
    expect(request?.path).toBe("/responses");
    expect(request?.headers["authorization"]).toBe(`Bearer ${token}`);
    expect(request?.headers["chatgpt-account-id"]).toBe("acct-1");
    expect(request?.headers["originator"]).toBe("pomerado");
    expect(request?.headers["accept"]).toBe("text/event-stream");
    expect(Object.keys(request?.headers ?? {}).some((name) => name.startsWith("x-stainless"))).toBe(false);
    expect(request?.body["store"]).toBe(false);
    expect(request?.body["stream"]).toBe(true);
    expect(request?.body["instructions"]).toBe("Answer briefly.");
    expect(request?.body["model"]).toBe("test-model");
  });

  it("maps model names and hoists developer instructions", () => {
    const body = toCodexRequest(
      {
        model: "asked-model",
        instructions: "",
        input: [
          {
            type: "message",
            role: "developer",
            content: [{ type: "input_text", text: "Rule one.", prompt_cache_breakpoint: { mode: "explicit" } }],
          },
          { type: "message", role: "developer", content: [{ type: "input_text", text: "Rule two." }] },
          { id: "msg_9", type: "message", role: "user", content: [{ type: "input_text", text: "hi", prompt_cache_breakpoint: {} }] },
        ],
        reasoning: { effort: "low", context: "all_turns", summary: "auto" },
        prompt_cache_options: { mode: "explicit" },
        max_output_tokens: 10,
        temperature: 0,
        store: true,
      },
      { "asked-model": "sent-model" },
    );
    expect(body).toEqual({
      model: "sent-model",
      instructions: "Rule one.\n\nRule two.",
      input: [{ type: "message", role: "user", content: [{ type: "input_text", text: "hi" }] }],
      reasoning: { effort: "low", summary: "auto" },
      store: false,
      stream: true,
    });
  });

  it("re-reads the login after a 401 and retries once with the newer token", async () => {
    const authPath = await writeAuth(freshToken("old"));
    const newer = freshToken("new");
    replies.push(() => {
      // The Codex CLI refreshed the login while the request was in flight.
      writeFileSync(
        authPath,
        JSON.stringify({ auth_mode: "chatgpt", tokens: { access_token: newer, account_id: "acct-1" } }),
      );
      return { status: 401, body: JSON.stringify({ detail: "expired" }), type: "application/json" };
    });
    replies.push(okText);
    const result = await run(chatgptSubscriptionProvider({ authPath, baseUrl: base }));
    expect(result.finalOutput).toBe("pong");
    expect(recorded).toHaveLength(2);
    expect(recorded[1]?.headers["authorization"]).toBe(`Bearer ${newer}`);
  });

  it("fails closed without retrying when a refused token has no replacement", async () => {
    const token = freshToken("only");
    const authPath = await writeAuth(token);
    replies.push(() => ({ status: 401, body: JSON.stringify({ detail: `bad ${token}` }), type: "application/json" }));
    const error = await run(chatgptSubscriptionProvider({ authPath, baseUrl: base })).catch((e: unknown) => e);
    expect(Reflect.get(Object(error), "status")).toBe(401);
    expect(Reflect.get(Object(error), "code")).toBe(new ChatgptSubscriptionAuthError().code);
    expect(String((error as Error).message)).not.toContain(token);
    expect(recorded).toHaveLength(1);
  });

  it("fails closed on an expired login in reread mode, before any request", async () => {
    const authPath = await writeAuth(staleToken("stale"));
    const error = await run(chatgptSubscriptionProvider({ authPath, baseUrl: base })).catch((e: unknown) => e);
    expect(Reflect.get(Object(error), "status")).toBe(401);
    expect(recorded).toHaveLength(0);
  });

  it("refreshes an expired login into its private store and leaves auth.json alone", async () => {
    const authPath = await writeAuth(staleToken("stale"));
    const before = await readFile(authPath, "utf8");
    const refreshed = freshToken("refreshed");
    const privateStorePath = join(dir, "private", "chatgpt-auth.json");
    replies.push((request) => {
      expect(request.path).toBe("/oauth/token");
      expect(request.body).toMatchObject({
        grant_type: "refresh_token",
        refresh_token: "refresh-token-original",
        client_id: expect.any(String),
      });
      return {
        status: 200,
        body: JSON.stringify({ access_token: refreshed, refresh_token: "refresh-token-rotated" }),
        type: "application/json",
      };
    });
    replies.push(okText);
    await run(
      chatgptSubscriptionProvider({
        authPath,
        baseUrl: base,
        tokenUrl: `${base}/oauth/token`,
        refresh: "private",
        privateStorePath,
      }),
    );
    expect(recorded[1]?.headers["authorization"]).toBe(`Bearer ${refreshed}`);
    expect(await readFile(authPath, "utf8")).toBe(before);
    const store = JSON.parse(await readFile(privateStorePath, "utf8")) as Record<string, unknown>;
    expect(store["refresh_token"]).toBe("refresh-token-rotated");
    expect((await stat(privateStorePath)).mode & 0o777).toBe(0o600);
  });

  it("reports a rate limit and a server error as the transient errors the retry check reads", async () => {
    const authPath = await writeAuth(freshToken("a"));
    const provider = chatgptSubscriptionProvider({ authPath, baseUrl: base });
    for (let attempt = 0; attempt < 3; attempt++)
      replies.push(() => ({
        status: 429,
        body: JSON.stringify({ error: { code: "rate_limit_exceeded", message: "slow down" } }),
        type: "application/json",
      }));
    const limited = await run(provider).catch((e: unknown) => e);
    expect(Reflect.get(Object(limited), "status")).toBe(429);
    expect(Reflect.get(Object(limited), "code")).toBe("rate_limit_exceeded");
    expect(providerQuotaExhausted(limited)).toBe(false);
    replies.length = 0;
    recorded.length = 0;
    for (let attempt = 0; attempt < 3; attempt++)
      replies.push(() => ({
        status: 200,
        body: sse({ type: "response.failed", response: { error: { code: "server_error", message: "boom" } } }),
      }));
    const failed = await run(provider).catch((e: unknown) => e);
    expect(Reflect.get(Object(failed), "status")).toBe(500);
    expect(Reflect.get(Object(failed), "code")).toBe("server_error");
  });

  it.each([
    { type: "response.failed", code: "usage_limit_reached", status: 429, expectedCode: "insufficient_quota" },
    { type: "error", code: "rate_limit_exceeded", status: 429, expectedCode: "rate_limit_exceeded" },
    { type: "response.error", code: "invalid_request_error", status: 400, expectedCode: "invalid_request_error" },
  ].flatMap((failure) => [false, true].map((stream) => ({ ...failure, stream }))))("preserves $type failure metadata (stream=$stream)", async ({ type, code, status, expectedCode, stream }) => {
    const token = freshToken("stream-error");
    const authPath = await writeAuth(token);
    const failure = { code, message: `rejected ${token}` };
    const attempts = !stream && code === "rate_limit_exceeded" ? 3 : 1;
    for (let attempt = 0; attempt < attempts; attempt++)
      replies.push(() => ({ status: 200, body: sse({ type, response: { error: failure }, ...failure }) }));
    const provider = chatgptSubscriptionProvider({ authPath, baseUrl: base });
    const error = await (async () => {
      if (!stream) return run(provider);
      const result = await new Runner({ modelProvider: provider, tracingDisabled: true }).run(
        new Agent({ name: "stream error probe", model: "test-model" }), "ping", { stream: true, maxTurns: 1 },
      );
      // Observe both promises so a failed stream never leaves an unhandled rejection.
      await Promise.all([result.completed, (async () => { for await (const _event of result) { /* drain */ } })()]);
    })().catch((error: unknown) => error);
    expect(error).toMatchObject({ status, code: expectedCode, name: "Error" });
    expect(error instanceof Error && error.message).not.toContain(token);
    expect(recorded).toHaveLength(attempts);
  });

  it("reports a spent subscription limit as exhausted quota, without retrying", async () => {
    const authPath = await writeAuth(freshToken("a"));
    replies.push(() => ({
      status: 429,
      body: JSON.stringify({ error: { type: "usage_limit_reached", message: "limit reached" } }),
      type: "application/json",
    }));
    const error = await run(chatgptSubscriptionProvider({ authPath, baseUrl: base })).catch((e: unknown) => e);
    expect(providerQuotaExhausted(error)).toBe(true);
    expect(recorded).toHaveLength(1);
  });

  it("answers a cache prewarm locally, since the backend has none", async () => {
    const authPath = await writeAuth(freshToken("a"));
    const provider = chatgptSubscriptionProvider({ authPath, baseUrl: base });
    const agent = new Agent({ name: "probe", instructions: "x", model: "test-model" });
    const usage = await prewarmInstructions(provider, 1, (prewarming) =>
      new Runner({ modelProvider: prewarming, tracingDisabled: true }).run(agent, "hi", { maxTurns: 1 }),
    );
    expect(usage.totalTokens).toBe(0);
    expect(recorded).toHaveLength(0);
  });

  it("redacts tokens and JWTs", () => {
    const token = freshToken("secret");
    const sample = ["rt", "123456789"].join("-");
    const text = redactSecrets(`Bearer abc.def ${token} ${JSON.stringify({ refresh_token: sample })} plain-secret-1`, [
      "plain-secret-1",
    ]);
    expect(text).not.toContain(token);
    expect(text).not.toContain("abc.def");
    expect(text).not.toContain(sample);
    expect(text).not.toContain("plain-secret-1");
  });

  it("reads expiry from the token", () => {
    expect(accessTokenExpiring(freshToken("a"), Date.now())).toBe(false);
    expect(accessTokenExpiring(staleToken("a"), Date.now())).toBe(true);
  });

  it("selects providers from the environment and fails closed on unknown values", () => {
    expect(modelProvidersFromEnv({})).toEqual({});
    const selected = modelProvidersFromEnv({ POMERADO_MODEL_TRANSPORT: "chatgpt-subscription", CODEX_HOME: dir });
    expect(selected.minterProvider).toBeDefined();
    expect(selected.guardianProvider).toBe(selected.minterProvider);
    expect(selected.outcomeReviewerProvider).toBe(selected.minterProvider);
    expect(() => modelProvidersFromEnv({ POMERADO_MODEL_TRANSPORT: "other" })).toThrow();
    expect(() =>
      modelProvidersFromEnv({ POMERADO_MODEL_TRANSPORT: "chatgpt-subscription", POMERADO_CODEX_MODEL_MAP: "[1]" }),
    ).toThrow();
  });
});
