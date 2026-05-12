/**
 * Layer 4: Effect Executor
 *
 * Spec:
 *   Layer 4: EFFECT EXECUTOR
 *   ├─ Impure boundary (network, filesystem, LLM, time)
 *   └─ Never modifies state; only executes intents
 *
 * Spec (Law 14 — Intent Deferred Execution):
 *   R(σ, ι) = [σ', I*]
 *   Guarantee: I* is emitted but NOT executed within R.
 *   Execution happens in Layer 4, outside reducer.
 *
 * LLM closure:
 *   executeIntents receives a stepFn callback from the runtime.
 *   When an LLM intent fires, Layer 4 calls Ollama, then calls stepFn with
 *   the response as an LLM_RESPONSE input.  That input goes through the full
 *   execution loop (Φ → checkpoint → hash-chain) — making AI output
 *   deterministic and replayable.
 */

import type { IntentList, LlmResponse } from "../layer3-intent/types"
import type { EffectHandlers, EffectResult } from "./types"
import { executeNetwork }        from "./network-effect"
import { executeStorage }        from "./storage-effect"
import { executeSchedule }       from "./schedule-effect"
import { executeLog, executeEmit } from "./logging-effect"
import { executeLLM }            from "./llm-effect"

export async function executeIntents(
  intents: IntentList,
  handlers: EffectHandlers,
  // Callback that feeds an LLM_RESPONSE back into the execution loop.
  // The runtime passes step() (partially applied) here.
  stepFn: (input: LlmResponse) => Promise<void>,
): Promise<EffectResult[]> {
  const results: EffectResult[] = []

  for (const intent of intents) {
    let result: EffectResult

    switch (intent.type) {
      case "SEND":
        result = await executeNetwork(intent, handlers.send)
        break
      case "STORE":
        result = await executeStorage(intent, handlers.store)
        break
      case "SCHEDULE":
        result = executeSchedule(intent, handlers.schedule)
        break
      case "LOG":
        result = executeLog(intent, handlers.log)
        break
      case "EMIT":
        result = executeEmit(intent, handlers.log)
        break
      case "LLM":
        // stepFn re-enters the loop with LLM_RESPONSE — see llm-effect.ts
        result = await executeLLM(intent, stepFn)
        break
      default: {
        const _exhaustive: never = intent
        result = { ok: false, intent: _exhaustive, error: new Error("Unknown intent type") }
      }
    }

    results.push(result)
  }

  return results
}
