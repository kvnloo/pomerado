/** Native Responses patch operations carried over a function-only subscription backend. */
const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const operationVariants = [
  { type: "create_file", move: false }, { type: "update_file", move: false },
  { type: "delete_file", move: false }, { type: "update_file", move: true },
].map(({ type, move }) => ({
  type: "object",
  properties: {
    type: { type: "string", enum: [type] },
    path: { type: "string" },
    ...(type === "delete_file" ? {} : { diff: { type: "string" } }),
    ...(move ? { move_to: { type: "string" } } : {}),
  },
  required: [...(type === "delete_file" ? ["type", "path"] : ["type", "path", "diff"]), ...(move ? ["move_to"] : [])],
  additionalProperties: false,
}));

export const codexPatchTool = {
  type: "function",
  name: "apply_patch",
  description: "Apply one file edit. Pass an operation with type create_file, update_file or delete_file, " +
    "and its path. For create_file, diff contains the file's lines prefixed with +. For update_file, " +
    "diff is a V4A patch body: @@ optionally followed by context, then space-prefixed unchanged lines, " +
    "-prefixed removed lines and +prefixed added lines. Separate hunks with @@. " +
    "Do not include *** Begin Patch, *** End Patch or file headers; path and type supply those. " +
    "For delete_file, omit diff.",
  parameters: {
    type: "object",
    properties: { operation: { anyOf: operationVariants } },
    required: ["operation"],
    additionalProperties: false,
  },
  strict: true,
};

/** Replays the SDK's native calls and results as function history, preserving call correlation. */
export const codexPatchInput = (item: unknown): unknown => {
  if (!record(item)) return item;
  if (item["type"] === "apply_patch_call")
    return {
      type: "function_call", name: "apply_patch", call_id: item["call_id"],
      arguments: JSON.stringify({ operation: item["operation"] }), status: item["status"],
    };
  if (item["type"] === "apply_patch_call_output")
    return {
      type: "function_call_output", call_id: item["call_id"],
      output: JSON.stringify({ status: item["status"] ?? "completed", output: item["output"] ?? "" }),
    };
  return item;
};

/** Reject malformed arguments before they can reach the host editor. */
export const nativePatchItem = (item: unknown): unknown => {
  if (!record(item) || item["type"] !== "function_call" || item["name"] !== "apply_patch") return item;
  let args: unknown;
  try {
    args = typeof item["arguments"] === "string" ? JSON.parse(item["arguments"]) : undefined;
  } catch {
    throw new Error("The apply_patch function returned invalid JSON arguments.");
  }
  const operation = record(args) ? args["operation"] : undefined;
  if (!record(operation) || typeof operation["path"] !== "string" ||
      !(operation["type"] === "delete_file" ||
        ((operation["type"] === "create_file" || operation["type"] === "update_file") &&
          typeof operation["diff"] === "string")))
    throw new Error("The apply_patch function returned an invalid operation.");
  return {
    type: "apply_patch_call", id: item["id"], call_id: item["call_id"],
    status: item["status"] ?? "completed", operation,
  };
};

export const nativePatchResponse = (response: unknown): unknown => {
  if (!record(response) || !Array.isArray(response["output"])) return response;
  return { ...response, output: response["output"].map(nativePatchItem) };
};
