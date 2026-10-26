/**
 * An opt-in model transport that sends Pomerado's model calls through a ChatGPT subscription
 * login, the one the Codex CLI keeps in `$CODEX_HOME/auth.json`, instead of an API key.
 *
 * It plugs in at the existing `ModelProvider` seam (`minterProvider`, `guardianProvider`,
 * `outcomeReviewerProvider`), so no minting or review code changes. The SDK's own Responses model
 * builds every request; this module only swaps the HTTP layer underneath it:
 *
 * - It adds the subscription's bearer token and account header, and never lets the token reach
 *   the SDK, a log or an error message.
 * - It reshapes the body for the subscription backend: non-empty `instructions`, `store: false`,
 *   streaming, no item IDs and only the body keys that backend takes.
 * - The backend only streams, so a non-streamed SDK call reads the stream to its end and gets
 *   the final response as JSON, as the Responses API would return it.
 * - Failures keep the shape the retry and quota checks read: `status`, `code` and the SDK's
 *   error names, with a spent subscription limit reported as `insufficient_quota`.
 *
 * Protocol details follow the open-source Codex CLI (openai/codex, Apache-2.0), used here only
 * as a reference.
 */
import { randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { mkdir, open, readFile, rename, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { OpenAIProvider, type ModelProvider } from "@openai/agents";
import OpenAI from "openai";
import { codexPatchInput, codexPatchTool, nativePatchItem, nativePatchResponse } from "./chatgpt-patch.js";

/** The value of `POMERADO_MODEL_TRANSPORT` that selects this transport. */
export const chatgptSubscriptionTransport = "chatgpt-subscription";

const defaultBaseUrl = "https://chatgpt.com/backend-api/codex";
const defaultTokenUrl = "https://auth.openai.com/oauth/token";
/** The public OAuth client of the Codex CLI, which issued the login this transport reuses. */
const codexClientId = "app_EMoamEEZ73f0CkXaXp7hrann";
const refreshSkewSeconds = 120;
const defaultInstructions = "You are a helpful assistant.";

/**
 * How the transport gets a fresh token once the current one is about to expire or is refused.
 * `reread` only reads `auth.json` again, so whatever keeps that file current, such as the Codex
 * CLI, stays the only writer; a stale token then fails closed. `private` refreshes the login
 * itself into a store of its own. Refresh tokens rotate, so that can sign the Codex CLI out.
 */
export type ChatgptRefreshMode = "reread" | "private";

export interface ChatgptSubscriptionOptions {
  /** Defaults to `$CODEX_HOME/auth.json`, or `~/.codex/auth.json`. */
  readonly authPath?: string;
  readonly refresh?: ChatgptRefreshMode;
  /** Where `private` refresh keeps its tokens. Defaults to `~/.pomerado/chatgpt-auth.json`. */
  readonly privateStorePath?: string;
  readonly baseUrl?: string;
  readonly tokenUrl?: string;
  /** Model names to send in place of the ones Pomerado asks for. */
  readonly modelMap?: Readonly<Record<string, string>>;
  /** The `originator` header, which names the calling app to the backend. */
  readonly originator?: string;
  /** A data residency requirement, sent as the Codex CLI sends its managed one. */
  readonly residency?: string;
  readonly fetch?: typeof globalThis.fetch;
  readonly now?: () => number;
}

/** A credential problem: the login is missing, of the wrong kind, stale or refused. Never retried. */
export class ChatgptSubscriptionAuthError extends Error {
  override readonly name = "ChatgptSubscriptionAuthError";
  readonly status = 401;
  readonly code = "chatgpt_subscription_auth";
}

interface Tokens {
  readonly accessToken: string;
  readonly accountId: string;
  readonly refreshToken: string | undefined;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const stringField = (value: unknown, key: string): string | undefined => {
  if (!isRecord(value)) return undefined;
  const field = value[key];
  return typeof field === "string" && field.length > 0 ? field : undefined;
};

/** Hides bearer tokens, JWTs and any of `secrets` that appear in `text`. */
export const redactSecrets = (text: string, secrets: readonly (string | undefined)[] = []) => {
  let out = text;
  for (const secret of secrets)
    if (secret !== undefined && secret.length >= 8) out = out.split(secret).join("[redacted]");
  return out
    .replace(/\beyJ[\w-]+\.[\w-]+\.[\w-]+/g, "[redacted]")
    .replace(/(bearer\s+)[^\s"',]+/gi, "$1[redacted]")
    .replace(/("(?:access_token|refresh_token|id_token)"\s*:\s*")[^"]*/g, "$1[redacted]");
};

const jwtClaims = (jwt: string): Record<string, unknown> | undefined => {
  const payload = jwt.split(".")[1];
  if (payload === undefined) return undefined;
  try {
    const claims: unknown = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
    return isRecord(claims) ? claims : undefined;
  } catch {
    return undefined;
  }
};

/** Whether `jwt` expires within `skewSeconds` of `nowMs`; a token without `exp` never does. */
export const accessTokenExpiring = (jwt: string, nowMs: number, skewSeconds = refreshSkewSeconds) => {
  const exp = jwtClaims(jwt)?.["exp"];
  return typeof exp === "number" && exp * 1000 - skewSeconds * 1000 <= nowMs;
};

const accountIdFromJwt = (jwt: string) =>
  stringField(jwtClaims(jwt)?.["https://api.openai.com/auth"], "chatgpt_account_id");

/** Reads the Codex CLI's login without changing it, and fails closed on anything unexpected. */
export const loadCodexAuth = async (path: string): Promise<Tokens> => {
  let text: string;
  try {
    text = await readFile(path, "utf8");
  } catch {
    throw new ChatgptSubscriptionAuthError(
      `No ChatGPT login found at ${path}. Sign in with the Codex CLI first.`,
    );
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new ChatgptSubscriptionAuthError(`The ChatGPT login at ${path} is not valid JSON.`);
  }
  if (stringField(parsed, "auth_mode") !== "chatgpt")
    throw new ChatgptSubscriptionAuthError(
      `The login at ${path} is not a ChatGPT subscription login.`,
    );
  const tokens = isRecord(parsed) ? parsed["tokens"] : undefined;
  const accessToken = stringField(tokens, "access_token");
  const accountId =
    stringField(tokens, "account_id") ??
    (accessToken === undefined ? undefined : accountIdFromJwt(accessToken));
  if (accessToken === undefined || accountId === undefined)
    throw new ChatgptSubscriptionAuthError(`The ChatGPT login at ${path} lacks a token or account.`);
  return { accessToken, accountId, refreshToken: stringField(tokens, "refresh_token") };
};

const defaultAuthPath = () =>
  join(process.env["CODEX_HOME"] ?? join(homedir(), ".codex"), "auth.json");
const defaultPrivateStorePath = () => join(homedir(), ".pomerado", "chatgpt-auth.json");

const readPrivateStore = async (path: string): Promise<Tokens | undefined> => {
  try {
    const parsed: unknown = JSON.parse(await readFile(path, "utf8"));
    const accessToken = stringField(parsed, "access_token");
    const accountId = stringField(parsed, "account_id");
    if (accessToken === undefined || accountId === undefined) return undefined;
    return { accessToken, accountId, refreshToken: stringField(parsed, "refresh_token") };
  } catch {
    return undefined;
  }
};

/** Holds `path.lock` while `work` runs, so two processes never refresh the same login at once. */
const withFileLock = async <T>(path: string, work: () => Promise<T>): Promise<T> => {
  const lock = `${path}.lock`;
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  const deadline = Date.now() + 30_000;
  for (;;) {
    try {
      const handle = await open(lock, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY, 0o600);
      await handle.close();
      break;
    } catch (error) {
      if (Reflect.get(Object(error), "code") !== "EEXIST" || Date.now() > deadline)
        throw new ChatgptSubscriptionAuthError("Could not lock the private ChatGPT token store.");
      await new Promise((resume) => setTimeout(resume, 50));
    }
  }
  try {
    return await work();
  } finally {
    await rm(lock, { force: true });
  }
};

interface TokenSourceOptions {
  readonly authPath: string;
  readonly refresh: ChatgptRefreshMode;
  readonly privateStorePath: string;
  readonly tokenUrl: string;
  readonly fetch: typeof globalThis.fetch;
  readonly now: () => number;
}

/** Supplies the current token, and a fresh one after the backend refused the last. */
class TokenSource {
  #current: Tokens | undefined;
  readonly #options: TokenSourceOptions;

  constructor(options: TokenSourceOptions) {
    this.#options = options;
  }

  /** Every secret this source has held, for redaction. */
  secrets(): readonly (string | undefined)[] {
    return [this.#current?.accessToken, this.#current?.refreshToken];
  }

  async current(): Promise<Tokens> {
    const cached = this.#current;
    if (cached !== undefined && !accessTokenExpiring(cached.accessToken, this.#options.now()))
      return cached;
    return this.#renew(cached?.accessToken);
  }

  /** The backend refused `used`: take a newer token if one exists, or fail closed. */
  async afterUnauthorized(used: string): Promise<Tokens> {
    return this.#renew(used, true);
  }

  async #renew(used: string | undefined, refused = false): Promise<Tokens> {
    const usable = (tokens: Tokens | undefined): tokens is Tokens =>
      tokens !== undefined &&
      tokens.accessToken !== (refused ? used : undefined) &&
      !accessTokenExpiring(tokens.accessToken, this.#options.now());
    const fromCli = await loadCodexAuth(this.#options.authPath);
    if (this.#options.refresh === "private") {
      const stored = await readPrivateStore(this.#options.privateStorePath);
      if (usable(stored)) return (this.#current = stored);
      if (usable(fromCli)) return (this.#current = fromCli);
      return (this.#current = await this.#refresh(stored ?? fromCli));
    }
    if (usable(fromCli)) return (this.#current = fromCli);
    throw new ChatgptSubscriptionAuthError(
      refused
        ? "The ChatGPT login was refused. Sign in again with the Codex CLI."
        : "The ChatGPT login has expired. Run the Codex CLI once to refresh it.",
    );
  }

  /** Refreshes into the private store, unless another process already did. */
  async #refresh(from: Tokens): Promise<Tokens> {
    const path = this.#options.privateStorePath;
    return withFileLock(path, async () => {
      const stored = await readPrivateStore(path);
      // Compare and swap: a process that refreshed while this one waited already rotated the
      // refresh token, so its result is the one to use.
      if (
        stored !== undefined &&
        stored.refreshToken !== from.refreshToken &&
        !accessTokenExpiring(stored.accessToken, this.#options.now())
      )
        return stored;
      const refreshToken = stored?.refreshToken ?? from.refreshToken;
      if (refreshToken === undefined)
        throw new ChatgptSubscriptionAuthError("The ChatGPT login has no refresh token.");
      let response: Response;
      try {
        response = await this.#options.fetch(this.#options.tokenUrl, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            client_id: codexClientId,
            grant_type: "refresh_token",
            refresh_token: refreshToken,
            scope: "openid profile email",
          }),
        });
      } catch {
        throw new ChatgptSubscriptionAuthError("Could not reach the ChatGPT sign-in service.");
      }
      if (!response.ok)
        throw new ChatgptSubscriptionAuthError(
          `The ChatGPT sign-in service refused the refresh (HTTP ${response.status}). Sign in again with the Codex CLI.`,
        );
      const body: unknown = await response.json().catch(() => undefined);
      const accessToken = stringField(body, "access_token");
      if (accessToken === undefined)
        throw new ChatgptSubscriptionAuthError("The ChatGPT sign-in service sent no access token.");
      const tokens: Tokens = {
        accessToken,
        accountId: accountIdFromJwt(accessToken) ?? from.accountId,
        refreshToken: stringField(body, "refresh_token") ?? refreshToken,
      };
      const temporary = `${path}.${randomUUID()}.tmp`;
      await writeFile(
        temporary,
        JSON.stringify({
          access_token: tokens.accessToken,
          account_id: tokens.accountId,
          refresh_token: tokens.refreshToken,
          last_refresh: new Date(this.#options.now()).toISOString(),
        }),
        { mode: 0o600 },
      );
      await rename(temporary, path);
      return tokens;
    });
  }
}

/** Body keys the subscription backend takes; any other key is dropped. */
const allowedBodyKeys = new Set([
  "model",
  "instructions",
  "input",
  "include",
  "tools",
  "tool_choice",
  "parallel_tool_calls",
  "reasoning",
  "text",
  "store",
  "stream",
  "prompt_cache_key",
  "service_tier",
  "context_management",
]);

const contentText = (content: unknown): string | undefined => {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return undefined;
  const parts = content.map((part) => stringField(part, "text"));
  return parts.every((part) => part !== undefined) ? parts.join("") : undefined;
};

/** An item's content parts without cache breakpoints, which the backend does not take. */
const withoutBreakpoints = (item: unknown): unknown => {
  if (!isRecord(item)) return item;
  if (item["type"] === "compaction")
    return { type: "compaction", encrypted_content: item["encrypted_content"] };
  const { id: _id, prompt_cache_breakpoint: _breakpoint, ...rest } = item;
  if (!Array.isArray(rest["content"])) return rest;
  return {
    ...rest,
    content: rest["content"].map((part) => {
      if (!isRecord(part)) return part;
      const { prompt_cache_breakpoint: _breakpoint, ...kept } = part;
      return kept;
    }),
  };
};

/**
 * Reshapes a Responses request body for the subscription backend: it needs non-empty top-level
 * `instructions`, so leading developer messages move back there; it stores nothing, so item IDs
 * would point at nothing; and it takes a narrower set of keys.
 */
export const toCodexRequest = (
  body: Record<string, unknown>,
  modelMap: Readonly<Record<string, string>> = {},
): Record<string, unknown> => {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(body)) if (allowedBodyKeys.has(key)) out[key] = value;
  const input =
    typeof body["input"] === "string"
      ? [{ type: "message", role: "user", content: [{ type: "input_text", text: body["input"] }] }]
      : Array.isArray(body["input"])
        ? [...body["input"]]
        : [];
  let instructions = typeof body["instructions"] === "string" ? body["instructions"] : "";
  if (instructions.trim() === "") {
    const hoisted: string[] = [];
    while (input.length > 0) {
      const first = input[0];
      const text =
        stringField(first, "role") === "developer" ? contentText(Reflect.get(Object(first), "content")) : undefined;
      if (text === undefined) break;
      hoisted.push(text);
      input.shift();
    }
    instructions = hoisted.join("\n\n");
  }
  out["instructions"] = instructions.trim() === "" ? defaultInstructions : instructions;
  out["input"] = input.map(codexPatchInput).map(withoutBreakpoints);
  if (Array.isArray(body["tools"]))
    out["tools"] = body["tools"].map((tool) =>
      isRecord(tool) && tool["type"] === "apply_patch" ? codexPatchTool : tool);
  if (isRecord(body["tool_choice"]) && body["tool_choice"]["type"] === "apply_patch")
    out["tool_choice"] = { type: "function", name: "apply_patch" };
  out["store"] = false;
  out["stream"] = true;
  if (isRecord(body["reasoning"])) {
    out["reasoning"] = Object.fromEntries(
      Object.entries(body["reasoning"]).filter(([key]) => key === "effort" || key === "summary"),
    );
  }
  const model = body["model"];
  if (typeof model === "string") out["model"] = modelMap[model] ?? model;
  return out;
};

const quotaCodes = new Set(["insufficient_quota", "usage_limit_reached", "usage_not_included"]);

/** An error response in the shape the OpenAI client turns into its own typed errors. */
const errorResponse = (
  status: number,
  error: { readonly message: string; readonly type?: string; readonly code?: string },
  final = false,
) => {
  const quota = quotaCodes.has(error.code ?? "") || quotaCodes.has(error.type ?? "");
  const body = {
    error: {
      message: error.message,
      type: error.type ?? null,
      code: quota ? "insufficient_quota" : (error.code ?? null),
      param: null,
    },
  };
  return new Response(JSON.stringify(body), {
    status: quota ? 429 : status,
    headers: {
      "content-type": "application/json",
      // A spent subscription limit or a refused login does not clear by retrying.
      ...(quota || final ? { "x-should-retry": "false" } : {}),
    },
  });
};

/** Maps a failed backend body, with any secret removed, to an OpenAI-style error response. */
const mappedFailure = (status: number, text: string, secrets: readonly (string | undefined)[]) => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = undefined;
  }
  const error = isRecord(parsed) && isRecord(parsed["error"]) ? parsed["error"] : parsed;
  const message = redactSecrets(
    stringField(error, "message") ?? stringField(parsed, "detail") ?? (text.slice(0, 500) || `HTTP ${status}`),
    secrets,
  );
  const code = stringField(error, "code");
  const type = stringField(error, "type");
  return errorResponse(status, {
    message,
    ...(code === undefined ? {} : { code }),
    ...(type === undefined ? {} : { type }),
  });
};

/** The HTTP status a `response.failed` error code stands for. */
const failedStatus = (code: string | undefined) => {
  if (code === undefined) return 500;
  if (code === "rate_limit_exceeded" || quotaCodes.has(code)) return 429;
  if (code.startsWith("invalid") || code.includes("context_length")) return 400;
  return 500;
};

/** Reads server-sent events from a response body. */
async function* sseEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<Record<string, unknown>> {
  const decoder = new TextDecoder();
  let buffer = "";
  const reader = body.getReader();
  const parse = (block: string) => {
    const data = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (data === "" || data === "[DONE]") return undefined;
    try {
      const event: unknown = JSON.parse(data);
      return isRecord(event) ? event : undefined;
    } catch {
      return undefined;
    }
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.search(/\r?\n\r?\n/);
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary).replace(/^\r?\n\r?\n/, "");
      const event = parse(block);
      if (event !== undefined) yield event;
      boundary = buffer.search(/\r?\n\r?\n/);
    }
  }
  const event = parse(buffer);
  if (event !== undefined) yield event;
}

/** Patch arguments arrive incrementally; expose an executable native item only when complete.
 * Failed terminal events become SDK API errors before its stream converter loses their codes.
 */
async function* patchEvents(body: ReadableStream<Uint8Array>, secrets: readonly (string | undefined)[], translatePatches: boolean) {
  const patchIds = new Set<unknown>();
  for await (const event of sseEvents(body)) {
    if (event["type"] === "response.failed" || event["type"] === "response.error" || event["type"] === "error") {
      const response = event["response"];
      const error = isRecord(response) && isRecord(response["error"]) ? response["error"]
        : isRecord(event["error"]) ? event["error"] : event;
      const failure = mappedFailure(failedStatus(stringField(error, "code") ?? stringField(error, "type")), JSON.stringify(error), secrets);
      throw OpenAI.APIError.generate(failure.status, await failure.json(), undefined, failure.headers);
    }
    if (!translatePatches) { yield event; continue; }
    const item = event["item"];
    if (isRecord(item) && item["type"] === "function_call" && item["name"] === "apply_patch") {
      patchIds.add(item["id"]);
      if (event["type"] === "response.output_item.added") continue;
      const native = nativePatchItem(item);
      yield { ...event, type: "response.output_item.added", item: native };
      yield { ...event, item: native };
    } else if (typeof event["type"] === "string" && event["type"].startsWith("response.function_call_arguments.") &&
        patchIds.has(event["item_id"])) {
      continue;
    } else if ("response" in event) {
      yield { ...event, response: nativePatchResponse(event["response"]) };
    } else {
      yield event;
    }
  }
}

const translatedPatchStream = (body: ReadableStream<Uint8Array>, secrets: readonly (string | undefined)[], translatePatches: boolean) => {
  const events = patchEvents(body, secrets, translatePatches);
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const next = await events.next();
        if (next.done) controller.close();
        else controller.enqueue(encoder.encode(`event: ${String(next.value["type"])}\ndata: ${JSON.stringify(next.value)}\n\n`));
      } catch (error) {
        controller.error(error);
      }
    },
    async cancel() { await events.return(); },
  });
};

/** Reads a streamed response to its end and answers as the non-streamed Responses API would. */
const collectStream = async (
  body: ReadableStream<Uint8Array>,
  secrets: readonly (string | undefined)[],
  translatePatches: boolean,
): Promise<Response> => {
  for await (const event of sseEvents(body)) {
    const type = event["type"];
    if (type === "response.completed" || type === "response.incomplete")
      return new Response(JSON.stringify(translatePatches ? nativePatchResponse(event["response"]) : event["response"]), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    if (type === "response.failed" || type === "response.error" || type === "error") {
      const response = event["response"];
      const error = isRecord(response) && isRecord(response["error"]) ? response["error"]
        : isRecord(event["error"]) ? event["error"] : event;
      const code = stringField(error, "code");
      const errorType = stringField(error, "type");
      return errorResponse(failedStatus(code ?? errorType), {
        message: redactSecrets(stringField(error, "message") ?? "The model response failed.", secrets),
        ...(code === undefined ? {} : { code }),
        ...(errorType === undefined ? {} : { type: errorType }),
      });
    }
  }
  // A stream that stops before its final event is a lost connection, which a retry can get past.
  return errorResponse(502, { message: "The model stream ended early.", code: "server_error" });
};

/** A prewarm only primes the provider's cache; the subscription backend has no such request. */
const isPrewarm = (body: Record<string, unknown>) => {
  const options = body["prompt_cache_options"];
  return isRecord(options) && options["prewarm"] === true;
};
const skippedPrewarm = (body: Record<string, unknown>) => ({
  id: `resp_prewarm_skipped_${randomUUID()}`,
  object: "response",
  created_at: 0,
  status: "completed",
  model: body["model"] ?? null,
  output: [],
  usage: {
    input_tokens: 0,
    input_tokens_details: { cached_tokens: 0 },
    output_tokens: 0,
    output_tokens_details: { reasoning_tokens: 0 },
    total_tokens: 0,
  },
});

/**
 * The `fetch` the OpenAI client calls: it signs each request with the subscription login and
 * reshapes requests and responses as the module comment describes.
 */
const makeChatgptFetch = (
  tokens: TokenSource,
  options: {
    readonly fetch: typeof globalThis.fetch;
    readonly modelMap: Readonly<Record<string, string>>;
    readonly originator: string;
    readonly residency: string | undefined;
    readonly sessionId: string;
  },
): typeof globalThis.fetch =>
  async (input, init) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const headers = new Headers(init?.headers);
    let body = init?.body;
    let wantsStream = true;
    let translatePatches = false;
    const isResponses = new URL(url).pathname.endsWith("/responses");
    if (typeof body === "string" && isResponses) {
      const parsed: unknown = JSON.parse(body);
      if (isRecord(parsed)) {
        wantsStream = parsed["stream"] === true;
        translatePatches = Array.isArray(parsed["tools"]) && parsed["tools"].some((tool) =>
          isRecord(tool) && tool["type"] === "apply_patch");
        if (isPrewarm(parsed)) {
          const skipped = skippedPrewarm(parsed);
          return wantsStream
            ? new Response(
                `event: response.completed\ndata: ${JSON.stringify({ type: "response.completed", response: skipped })}\n\n`,
                { status: 200, headers: { "content-type": "text/event-stream" } },
              )
            : new Response(JSON.stringify(skipped), {
                status: 200,
                headers: { "content-type": "application/json" },
              });
        }
        // Never drop a checkpoint or pretend a compaction trigger succeeded: the minter
        // can disable its optional background compaction when this route refuses it.
        const unsupported = Array.isArray(parsed["input"]) && parsed["input"].find((item) =>
          isRecord(item) && (item["type"] === "compaction_trigger" || item["type"] === "item_reference"));
        if (isRecord(unsupported))
          return errorResponse(400, {
            message: `The subscription transport does not support ${String(unsupported["type"])}; send full history and inline items.`,
            code: "unsupported_input_item", type: "invalid_request_error",
          }, true);
        body = JSON.stringify(toCodexRequest(parsed, options.modelMap));
      }
    }
    for (const name of [...headers.keys()])
      if (name.startsWith("x-stainless") || name === "openai-organization" || name === "openai-project")
        headers.delete(name);
    headers.set("originator", options.originator);
    headers.set("user-agent", `${options.originator} (pomerado; chatgpt-subscription)`);
    headers.set("session_id", options.sessionId);
    if (isResponses) headers.set("accept", "text/event-stream");
    if (options.residency !== undefined)
      headers.set("x-openai-internal-codex-residency", options.residency);
    const send = (login: Tokens) => {
      headers.set("authorization", `Bearer ${login.accessToken}`);
      headers.set("chatgpt-account-id", login.accountId);
      return options.fetch(url, { ...init, headers, ...(body === undefined ? {} : { body }) });
    };
    const signedSend = async (login: Tokens) => {
      try {
        return await send(login);
      } catch (error) {
        // Keep the socket code a retry check reads, and nothing that could carry the token.
        const failure = new Error(
          redactSecrets(error instanceof Error ? error.message : "The model request failed.", tokens.secrets()),
        );
        const cause: unknown = error instanceof Error ? error.cause : undefined;
        const code = Reflect.get(Object(cause), "code") ?? Reflect.get(Object(error), "code");
        if (code !== undefined) Object.assign(failure, { code });
        failure.name = error instanceof Error && error.name === "AbortError" ? "AbortError" : failure.name;
        throw failure;
      }
    };
    // A login problem answers as a refused request, not a thrown error: the OpenAI client would
    // report a thrown one as a lost connection, which the retry check takes as transient.
    const refusedLogin = (error: unknown) => {
      if (!(error instanceof ChatgptSubscriptionAuthError)) throw error;
      return errorResponse(401, { message: error.message, code: error.code }, true);
    };
    let login: Tokens;
    try {
      login = await tokens.current();
    } catch (error) {
      return refusedLogin(error);
    }
    let response = await signedSend(login);
    if (response.status === 401) {
      await response.body?.cancel();
      try {
        login = await tokens.afterUnauthorized(login.accessToken);
      } catch (error) {
        return refusedLogin(error);
      }
      response = await signedSend(login);
    }
    if (!response.ok) return mappedFailure(response.status, await response.text(), tokens.secrets());
    if (!isResponses || response.body === null) return response;
    if (wantsStream) return new Response(translatedPatchStream(response.body, tokens.secrets(), translatePatches), {
      status: response.status, headers: response.headers,
    });
    return collectStream(response.body, tokens.secrets(), translatePatches);
  };

/**
 * A `ModelProvider` that sends model calls through the ChatGPT subscription login. One provider
 * can serve the minter, the Guardian and the outcome reviewer.
 */
export const chatgptSubscriptionProvider = (options: ChatgptSubscriptionOptions = {}): ModelProvider => {
  const doFetch = options.fetch ?? globalThis.fetch;
  const tokens = new TokenSource({
    authPath: options.authPath ?? defaultAuthPath(),
    refresh: options.refresh ?? "reread",
    privateStorePath: options.privateStorePath ?? defaultPrivateStorePath(),
    tokenUrl: options.tokenUrl ?? defaultTokenUrl,
    fetch: doFetch,
    now: options.now ?? Date.now,
  });
  const modelMap = options.modelMap ?? {};
  const client = new OpenAI({
    // The real token is added per request below; the client never holds it.
    apiKey: "chatgpt-subscription",
    baseURL: options.baseUrl ?? defaultBaseUrl,
    fetch: makeChatgptFetch(tokens, {
      fetch: doFetch,
      modelMap,
      originator: options.originator ?? "pomerado",
      residency: options.residency,
      sessionId: randomUUID(),
    }),
  });
  const inner = new OpenAIProvider({ openAIClient: client, useResponses: true });
  return {
    getModel: (name?: string) => inner.getModel(name === undefined ? name : (modelMap[name] ?? name)),
  };
};

const parseModelMap = (text: string | undefined): Record<string, string> => {
  if (text === undefined || text.trim() === "") return {};
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("POMERADO_CODEX_MODEL_MAP must be a JSON object of model names.");
  }
  if (!isRecord(parsed) || !Object.values(parsed).every((value) => typeof value === "string"))
    throw new Error("POMERADO_CODEX_MODEL_MAP must be a JSON object of model names.");
  return parsed as Record<string, string>;
};

/**
 * The model providers the environment selects for the CLIs. Unset `POMERADO_MODEL_TRANSPORT`
 * keeps the default OpenAI API key route; an unknown value fails closed.
 */
export const modelProvidersFromEnv = (
  env: NodeJS.ProcessEnv = process.env,
): {
  readonly minterProvider?: ModelProvider;
  readonly guardianProvider?: ModelProvider;
  readonly outcomeReviewerProvider?: ModelProvider;
} => {
  const transport = env["POMERADO_MODEL_TRANSPORT"];
  if (transport === undefined || transport === "" || transport === "openai-api") return {};
  if (transport !== chatgptSubscriptionTransport)
    throw new Error(
      `Unknown POMERADO_MODEL_TRANSPORT. Use ${chatgptSubscriptionTransport}, or leave it unset.`,
    );
  const refresh = env["POMERADO_CHATGPT_REFRESH"] ?? "reread";
  if (refresh !== "reread" && refresh !== "private")
    throw new Error("POMERADO_CHATGPT_REFRESH must be reread or private.");
  const residency = env["POMERADO_CODEX_RESIDENCY"];
  const provider = chatgptSubscriptionProvider({
    refresh,
    modelMap: parseModelMap(env["POMERADO_CODEX_MODEL_MAP"]),
    ...(env["CODEX_HOME"] === undefined ? {} : { authPath: join(env["CODEX_HOME"], "auth.json") }),
    ...(residency === undefined || residency === "" ? {} : { residency }),
  });
  return { minterProvider: provider, guardianProvider: provider, outcomeReviewerProvider: provider };
};
