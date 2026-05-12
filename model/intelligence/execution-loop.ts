/**
 * Runtime — Execution Loop
 *
 * Spec (Single-Node Deterministic Execution):
 *   loop:
 *     input ← read from queue
 *     [state, intents] ← Φ(state, input)
 *     checkpoint.record({ before, after: state, intents })
 *     await executeIntents(intents)
 *     intents ← []
 *
 * Property: Deterministic. No race conditions, no nondeterminism.
 *
 * Spec Law 12 (Replay Theorem):
 *   Replaying log + reducers → same sequence of states
 *   σᵢ = event[i].after ∀i
 *
 * Spec Law 13 (Hash Chain Integrity):
 *   hash[i]      = SHA256(event[i] with {hash: undefined, prevHash: undefined})
 *   event[i].prevHash = hash[i-1]
 *
 * LLM closure (deterministic AI):
 *   When Φ emits an LLM intent, executeIntents calls Ollama (Layer 4),
 *   then calls step() again with the LLM_RESPONSE as a normal input.
 *   That second step is fully logged, hashed, and checkpointed.
 *   On replay, LLM_RESPONSE events are replayed from the log verbatim —
 *   Ollama is never called — so replay remains deterministic.
 *
 *   Flow:
 *     step(σ, userInput)
 *       → Φ emits { type: "LLM", requestId, ... }
 *       → executeIntents calls Ollama
 *       → Ollama responds
 *       → step(σ', { type: "LLM_RESPONSE", requestId, response })
 *           → Φ processes response
 *           → checkpointed, hash-chained
 */

import { createHash } from "crypto"
import type { Reducer, CheckpointEvent, HLC, RuntimeConfig } from "./types"
import type { IntentList, LlmResponse } from "../layer3-intent/types"
import { executeIntents } from "../layer4-effects/effect-executor"

// ─── HLC tick ─────────────────────────────────────────────────────────────────
//
// Spec Type 8: HLC = { logical, physical, nodeId }
// Time access is isolated here — reducers (Φ) never see the clock.

export function tickHLC(prev: HLC, nodeId: string): HLC {
  const physical = Date.now()
  return {
    logical:  prev.logical + 1,
    physical: Math.max(physical, prev.physical),
    nodeId,
  }
}

// ─── Hash chain ───────────────────────────────────────────────────────────────

export function hashEvent<Σ>(
  event: Omit<CheckpointEvent<Σ>, "hash" | "prevHash"> & { prevHash: string }
): string {
  const payload = JSON.stringify({ ...event, hash: undefined })
  return createHash("sha256").update(payload).digest("hex")
}

// ─── Single execution step ────────────────────────────────────────────────────

export type StepResult<Σ> = {
  state:   Σ
  intents: IntentList
  event:   CheckpointEvent<Σ>
}

export async function step<Σ>(
  state:    Σ,
  input:    unknown,
  phi:      Reducer<Σ>,
  hlc:      HLC,
  nodeId:   string,
  prevHash: string,
  effects:  RuntimeConfig<Σ>["effects"],
  // Mutable log ref — the runtime passes its checkpoint array so that
  // LLM_RESPONSE steps (which call step() recursively from Layer 4) are
  // automatically appended and their prevHash is correct.
  log:      CheckpointEvent<Σ>[],
): Promise<StepResult<Σ>> {
  const before = state

  const [after, intents] = phi(state, input)

  const timestamp = tickHLC(hlc, nodeId)

  const partial = {
    nodeId,
    timestamp,
    type:     "REDUCE" as const,
    before,
    after,
    intents,
    prevHash,
  }
  const hash  = hashEvent(partial)
  const event: CheckpointEvent<Σ> = { ...partial, hash }

  // Append to log before executing intents so that any recursive step()
  // call (from LLM_RESPONSE) gets the correct prevHash.
  log.push(event)

  // Build the stepFn closure that Layer 4 will call with the LLM response.
  // It re-enters step() with the LLM_RESPONSE as a normal input — meaning
  // the response is checkpointed and hash-chained exactly like everything else.
  const stepFn = async (llmResponse: LlmResponse): Promise<void> => {
    const lastEvent = log[log.length - 1]
    await step(
      after,             // state after the step that emitted the LLM intent
      llmResponse,       // { type: "LLM_RESPONSE", requestId, model, response }
      phi,
      timestamp,         // HLC from this step — LLM_RESPONSE advances it further
      nodeId,
      lastEvent?.hash ?? "",
      effects,
      log,
    )
  }

  await executeIntents(intents, effects, stepFn)

  return { state: after, intents, event }
}

// ─── Replay ───────────────────────────────────────────────────────────────────
//
// Spec Law 12: Given log + initial state + reducers → reconstruct state.
// Re-executes Φ for each event; asserts reconstructed state matches event.after.
// Does NOT re-execute intents (replay is read-only re-derivation).
// LLM_RESPONSE events replay from the log — Ollama is never called.

export async function replayLog<Σ>(
  log:          CheckpointEvent<Σ>[],
  initialState: Σ,
  phi:          Reducer<Σ>,
  eq:           (a: Σ, b: Σ) => boolean,
): Promise<{ ok: boolean; failIndex: number | null; states: Σ[] }> {
  let state    = initialState
  const states: Σ[] = [state]
  let prevHash = ""

  for (let i = 0; i < log.length; i++) {
    const event = log[i]

    if (event.prevHash !== prevHash) {
      return { ok: false, failIndex: i, states }
    }

    const expectedHash = hashEvent({
      nodeId:    event.nodeId,
      timestamp: event.timestamp,
      type:      event.type,
      before:    event.before,
      after:     event.after,
      intents:   event.intents,
      prevHash:  event.prevHash,
    })
    if (expectedHash !== event.hash) {
      return { ok: false, failIndex: i, states }
    }

    const [derived] = phi(state, undefined)
    if (!eq(derived, event.after as Σ)) {
      return { ok: false, failIndex: i, states }
    }

    state    = event.after as Σ
    prevHash = event.hash
    states.push(state)
  }

  return { ok: true, failIndex: null, states }
}
