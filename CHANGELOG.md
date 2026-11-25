# Changelog

## 0.2.1

- The agent that receives a handoff is reminded to answer the user. Without it, a model often replied to the transfer ("Transferred.") or with nothing at all, because its last input was the handoff's tool result rather than a user message. The reminder is a system message appended as the last message of that agent's first model call, for that call only and never saved to the thread. Mastra's per-call `system` and `context` options were not enough on their own: both go before the history, where the tool result still reads as the last word.
- `handoffPrompt` on `createSwarm` replaces the reminder (`({ from, to, target }) => string`) or turns it off (`false`). `defaultHandoffPrompt` is exported.

## 0.2.0

- Nested swarms: a swarm can be a member of another swarm. A nested swarm needs a `name`, which is its handoff target. Each level remembers its own active member, follow-ups go straight to the active agent at any depth, and coming back to a nested swarm resumes its last active agent.
- Names must be unique across the whole tree, and any agent can hand off to any name in it.
- `activePath` on results and stream output, `getActivePath` on the swarm, `path` on `swarm-agent-start`, and `target` on a handoff into a nested swarm.
- A custom `ActiveAgentStore` receives a `scope`: undefined for the top-level swarm, the nested swarm's name below it.
- Single-level swarms behave as in 0.1, and keep storing `activeAgent` in thread metadata.

## 0.1.1

- README: the end-of-reply marker after a handoff is only needed before `@mastra/core` 1.76, which fixed steps that start with a tool call being merged into the previous step. No behaviour changes.

## 0.1.0

First release: swarm-style handoffs for Mastra, ported from LangGraph Swarm.

- `createHandoffTool` makes `transfer_to_<agent>` tools that hand the conversation to another agent.
- `createSwarm` keeps each thread's active agent in its memory metadata, or in a store you provide, and sends every turn straight to that agent.
- `generate` and `stream`; `stream` adds `swarm-agent-start` and `swarm-handoff` events to Mastra's chunks.
- `getActiveAgent` and `setActiveAgent` to read or move a thread's active agent.
- `maxHandoffs` stops agents passing a turn back and forth forever.
- Each agent's reply is stored as its own assistant message, so a handoff never reads to the model as one parallel step.
- Tested on `@mastra/core` 1.63 and the latest 1.x.

## 0.0.1

Published by hand to create the package on npm, so trusted publishing could be set up. Same code as 0.1.0, without provenance.
