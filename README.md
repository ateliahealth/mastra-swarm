# @atelia/mastra-swarm

Swarm-style multi-agent handoffs for [Mastra](https://mastra.ai), ported from [LangGraph Swarm](https://github.com/langchain-ai/langgraphjs/tree/main/libs/langgraph-swarm).

In a swarm, agents hand the conversation to each other with `transfer_to_<agent>` tools, and each thread remembers which agent is active. The next user message goes straight to that agent: no supervisor routes it, and no supervisor rewrites the answer.

Mastra's built-in multi-agent patterns are supervisors (`agents: {...}` on an agent) and `agent.network()`. Both run a routing model on every turn. A swarm removes that call whenever the conversation stays with the same specialist, which in most chat products is most follow-ups.

## Install

```bash
npm install @atelia/mastra-swarm
```

Requires `@mastra/core` 1.63 or newer (tested on 1.63 and 1.85) and `zod` 3.25+ or 4.

## Usage

```ts
import { Agent } from '@mastra/core/agent';
import { createTool } from '@mastra/core/tools';
import { Memory } from '@mastra/memory';
import { LibSQLStore } from '@mastra/libsql';
import { openai } from '@ai-sdk/openai';
import { z } from 'zod';
import { createHandoffTool, createSwarm } from '@atelia/mastra-swarm';

const model = openai('gpt-4o');
const memory = new Memory({ storage: new LibSQLStore({ url: 'file:swarm.db' }) });

const add = createTool({
  id: 'add',
  description: 'Add two numbers',
  inputSchema: z.object({ a: z.number(), b: z.number() }),
  execute: async ({ a, b }) => a + b,
});

const transferToBob = createHandoffTool({ agentName: 'Bob' });
const transferToAlice = createHandoffTool({
  agentName: 'Alice',
  description: 'Transfer to Alice, she can help with math',
});

const alice = new Agent({
  id: 'alice',
  name: 'Alice',
  instructions: 'You are Alice, an addition expert.',
  model,
  memory,
  tools: { add, [transferToBob.id]: transferToBob },
});

const bob = new Agent({
  id: 'bob',
  name: 'Bob',
  instructions: 'You are Bob, you speak like a pirate.',
  model,
  memory,
  tools: { [transferToAlice.id]: transferToAlice },
});

const swarm = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice' });
const conversation = { memory: { thread: 'thread-1', resource: 'user-1' } };

await swarm.generate("i'd like to speak to Bob", conversation);
// Alice calls transfer_to_bob, Bob answers like a pirate. Bob is now active.

await swarm.generate("what's 5 + 7?", conversation);
// Goes straight to Bob. Bob hands back to Alice, who calls add and answers 12.
```

Run it yourself with `OPENAI_API_KEY=... pnpm example` (or the `AZURE_OPENAI_*` variables in [examples/alice-bob.ts](examples/alice-bob.ts)).

### Streaming

```ts
const output = swarm.stream('hello', conversation);
for await (const chunk of output.fullStream) {
  if (chunk.type === 'swarm-agent-start') console.log(`\n${chunk.payload.agentName}:`);
  if (chunk.type === 'swarm-handoff') console.log(`→ ${chunk.payload.to}`);
  if (chunk.type === 'text-delta') process.stdout.write(chunk.payload.text);
}
console.log(await output.activeAgent);
```

`fullStream` carries every Mastra chunk from every agent in the turn, plus `swarm-agent-start` and `swarm-handoff` events. `textStream`, `text`, `activeAgent`, `activePath` and `handoffs` are there too.

### Nested swarms

A swarm can be a member of another swarm. Give the inner swarm a `name`, and other agents hand off to it by that name:

```ts
const workflows = createSwarm({
  name: 'workflows',
  agents: [workflowManager, intake, forms],
  defaultActiveAgent: 'workflow-manager',
});

const clinic = createSwarm({ agents: [frontDesk, workflows], defaultActiveAgent: 'frontdesk' });
```

- `frontDesk` hands off with `createHandoffTool({ agentName: 'workflows' })`, and the conversation goes to the workflows swarm's active agent, or `workflow-manager` the first time.
- Agents inside `workflows` hand off to each other, and leave by handing off to an agent outside it, such as `frontdesk`.
- Each level remembers its own active member. A follow-up goes straight to the active agent however deep it is, and coming back to `workflows` resumes the agent that was last active there.
- Names are unique across the whole tree, and any agent can hand off to any name in it, including an agent inside a nested swarm (`transfer_to_intake` from the front desk).
- `activePath` and the `path` on `swarm-agent-start` give the names from the top down to the active agent, e.g. `['workflows', 'intake']`.

See [examples/nested.ts](examples/nested.ts).

## API

**`createHandoffTool({ agentName, name?, description? })`** returns a Mastra tool named `transfer_to_<agent_name>` (lowercased, whitespace to `_`) that hands the conversation to `agentName`. The default description is `Ask agent '<agentName>' for help`.

**`createSwarm({ agents, defaultActiveAgent, name?, store?, maxHandoffs? })`** returns a swarm. `agents` takes agents and named swarms. It has:

- `generate(messages, { memory: { thread, resource }, activeAgent?, requestContext? })` resolves to `{ text, activeAgent, activePath, handoffs }`. A handoff to a nested swarm is reported as `{ from, to, target }`, where `to` is the agent that answered and `target` the swarm's name.
- `stream(messages, options)` returns the streaming output above.
- `getActiveAgent({ thread, resource })`, `getActivePath(...)` and `setActiveAgent({ thread, resource }, name)` read or move a thread's active agent, for example to send a user back to a front-desk agent. `name` can be an agent at any level, or a nested swarm.

A custom `store` gets `get(ref, scope?)` and `set(ref, agentName, scope?)`, where `scope` is undefined for the top-level swarm and a nested swarm's name below it. The default store keeps them in thread metadata as `activeAgent` and `activeAgent:<swarm name>`.

Pass `activeAgent` to start one turn with a particular agent. `maxHandoffs` (default 10) caps handoffs within a single turn, so two agents passing the conversation back and forth fail loudly instead of looping.

**`getHandoffDestinations(agent)`** lists the agents an agent can hand off to. **`getHandoffDestination(tool)`** returns a single handoff tool's target.

## How it works

| LangGraph Swarm                                | mastra-swarm                                                      |
| ---------------------------------------------- | ----------------------------------------------------------------- |
| `activeAgent` in graph state, kept by a checkpointer | `activeAgent` in the Mastra thread's metadata (or your `store`) |
| Router edge from `START`                       | `createSwarm` reads the thread's active agent before each turn    |
| Handoff tool returns `Command({ goto })`       | Handoff tool flags the target in the request context; `stopWhen` ends the agent's run there |
| Shared `messages` state                        | Every agent uses the same memory thread                           |

After a handoff, the swarm marks the handing-off agent's last message as the end of its response, so the next agent's reply is stored as its own assistant message. Mastra before 1.76 folded a step that starts with a tool call into the previous step ([#26685](https://github.com/mastra-ai/mastra/pull/26685)), so the next agent's first tool call read to the model as a parallel call of the agent that handed off. On 1.76 and later the marker is not needed for that, and it does no harm.

## Requirements and limits

- All agents in a swarm must share one memory (the same storage), since the swarm keeps one conversation per thread.
- Two requests on the same thread at the same time can race on the active agent, as they would on any thread.
- Messages are not tagged with the agent that wrote them yet; LangGraph sets `name` on each AI message.

## Development

```bash
pnpm install
pnpm test        # offline, with scripted mock models
pnpm typecheck
pnpm build
```

The tests port every case from LangGraph Swarm's JavaScript and Python suites, plus the behaviour their code has that those suites don't cover.

## Releasing

Bump `version` in `package.json`, merge it, then push a matching tag (`v0.1.0` for version `0.1.0`). The [Publish workflow](.github/workflows/publish.yml) checks the tag against the version, runs the checks, and publishes to npm through trusted publishing with provenance. No npm token is stored anywhere.

## License

MIT. Derived from LangGraph Swarm, © LangChain, MIT.
