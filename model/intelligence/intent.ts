/**
 * Layer 3: Intent Stream — Intent Constructors
 *
 * Spec: Intent is a first-class value, not an operation.
 * Reducers emit intents; Layer 4 executes them.
 * Invariant: Reducers can emit intents; intents are never interpreted by reducers.
 */

import type { Intent } from "./types"

export function send(to: string, opcode: number, payload: unknown): Intent {
  return { type: "SEND", to, opcode, payload }
}

export function store(key: string, value: unknown): Intent {
  return { type: "STORE", key, value }
}

export function schedule(reducerId: string, delayMs: number): Intent {
  return { type: "SCHEDULE", reducerId, delayMs }
}

export function log(level: "info" | "warn" | "error", msg: string): Intent {
  return { type: "LOG", level, msg }
}

export function emitIntent(channel: string, payload: unknown): Intent {
  return { type: "EMIT", channel, payload }
}

/**
 * Emit an LLM intent.
 *
 * @param requestId - Caller-supplied correlation ID. The matching LLM_RESPONSE
 *   input will carry the same ID, letting the reducer route the response back
 *   to the correct piece of state without any mutable bookkeeping.
 *
 * Usage:
 *   import { randomUUID } from "crypto"
 *   return [newState, [llm(randomUUID(), "mistral", "Summarise: ...", 512)]]
 */
export function llm(
  requestId: string,
  model:     string,
  prompt:    string,
  maxTokens: number,
): Intent {
  return { type: "LLM", requestId, model, prompt, maxTokens }
}
