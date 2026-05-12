/**
 * Layer 4: Effect Executor — LLM Effect (Ollama / Mistral)
 *
 * Spec Intent variant:
 *   { type: "LLM"; requestId: string; model: string; prompt: string; maxTokens: number }
 *
 * Invariant: Never modifies state Σ.
 *
 * Closure loop:
 *   1. Reducer emits { type: "LLM", requestId, model: "mistral", prompt, maxTokens }
 *   2. executeLLM() calls Ollama at http://localhost:11434/api/generate
 *   3. Streams and collects the full response text
 *   4. Calls stepFn({ type: "LLM_RESPONSE", requestId, model, response })
 *      — this re-enters the execution loop, checkpointing the response as a
 *        normal input.  From here it is hash-chained and replayable.
 *   5. On replay, LLM_RESPONSE events come from the checkpoint log — Ollama
 *      is never called again, so replay is fully deterministic.
 *
 * Configuration:
 *   OLLAMA_URL  — defaults to http://localhost:11434/api/generate
 */

import type { Intent, LlmResponse } from "../layer3-intent/types"
import type { EffectResult } from "./types"

const OLLAMA_URL =
  process.env["OLLAMA_URL"] ?? "http://localhost:11434/api/generate"

// ─── Ollama streaming client ──────────────────────────────────────────────────

async function callOllama(
  model:     string,
  prompt:    string,
  maxTokens: number,
): Promise<string> {
  const res = await fetch(OLLAMA_URL, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      prompt,
      options: { num_predict: maxTokens },
      stream:  true,
    }),
  })

  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Ollama ${res.status}: ${body}`)
  }

  if (!res.body) throw new Error("Ollama returned no body")

  // Stream NDJSON lines — each line is { model, response, done, ... }
  const reader  = res.body.getReader()
  const decoder = new TextDecoder()
  let   buf     = ""
  let   text    = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const lines = buf.split("\n")
    buf = lines.pop() ?? ""        // keep any incomplete trailing line
    for (const line of lines) {
      if (!line.trim()) continue
      try {
        const obj = JSON.parse(line) as { response?: string; done?: boolean }
        text += obj.response ?? ""
        if (obj.done) break
      } catch {
        // malformed line — skip
      }
    }
  }

  return text.trim()
}

// ─── Effect executor ──────────────────────────────────────────────────────────

/**
 * @param intent   - The LLM intent emitted by a reducer.
 * @param stepFn   - Callback that feeds the response back into the execution
 *                   loop as an LLM_RESPONSE input.  Supplied by the runtime;
 *                   called once after Ollama responds.
 */
export async function executeLLM(
  intent:  Extract<Intent, { type: "LLM" }>,
  stepFn:  (input: LlmResponse) => Promise<void>,
): Promise<EffectResult> {
  try {
    const response = await callOllama(
      intent.model,
      intent.prompt,
      intent.maxTokens,
    )

    // Re-enter the execution loop — this is what makes AI deterministic.
    // The response string becomes a normal reducer input: logged, hashed,
    // checkpointed.  Replay replays the LLM_RESPONSE from the log;
    // Ollama is never called again.
    await stepFn({
      type:      "LLM_RESPONSE",
      requestId: intent.requestId,
      model:     intent.model,
      response,
    })

    return { ok: true, intent }
  } catch (error) {
    return { ok: false, intent, error }
  }
}
