/**
 * Layer 3: Intent Stream — Types
 *
 * Spec: Intent = { type: T, payload: P }
 * Spec: IntentList = [I₁, I₂, ..., Iₙ] — Free Monoid
 *
 * Properties (from spec):
 * - Opaque to reducer algebra (reducers can only emit, never interpret)
 * - Concatenable: [] is unit, ++ is associative
 * - Replayable: same input + same reducer = same intents
 * - Deferrable: emission ≠ execution
 *
 * LLM loop closure:
 *   Reducers emit { type: "LLM", requestId, ... }.
 *   Layer 4 calls Ollama, then re-enters the execution loop via
 *   step(state, { type: "LLM_RESPONSE", requestId, response }, ...).
 *   The response is therefore hash-chained and replayable like any other input.
 *   On replay, LLM_RESPONSE events are replayed verbatim from the log —
 *   the model is never called again.
 */

export type Intent =
  | { type: "SEND";     to: string; opcode: number; payload: unknown }
  | { type: "STORE";    key: string; value: unknown }
  | { type: "SCHEDULE"; reducerId: string; delayMs: number }
  | { type: "LOG";      level: "info" | "warn" | "error"; msg: string }
  | { type: "EMIT";     channel: string; payload: unknown }
  | {
      type:      "LLM"
      requestId: string   // Correlates the intent to its LLM_RESPONSE input
      model:     string
      prompt:    string
      maxTokens: number
    }

export type IntentList = readonly Intent[]

// ─── LLM_RESPONSE ─────────────────────────────────────────────────────────────
//
// Not an Intent — this is an *input* to Φ.  Layer 4 constructs it after
// Ollama returns and passes it to step() as a normal reducer input.
// From that point on it is indistinguishable from any other input:
// logged, hashed, checkpointed, replayable.

export type LlmResponse = {
  type:      "LLM_RESPONSE"
  requestId: string   // Matches the LLM intent that triggered this call
  model:     string
  response:  string
}
