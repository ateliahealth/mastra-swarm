import type { Agent } from '@mastra/core/agent';
import type { MastraMemory } from '@mastra/core/memory';
import { RequestContext } from '@mastra/core/request-context';
import type { ChunkType } from '@mastra/core/stream';
import { METADATA_KEY_HANDOFF_DESTINATION, getHandoffDestinations } from './handoff';

type AnyAgent = Agent<any, any, any>;
type MessageInput = Parameters<AnyAgent['stream']>[0];

export interface ThreadRef {
  thread: string;
  resource: string;
}

/** Where the swarm remembers each thread's active agent. Defaults to the thread's metadata. */
export interface ActiveAgentStore {
  get(ref: ThreadRef): Promise<string | undefined>;
  set(ref: ThreadRef, agentName: string): Promise<void>;
}

export interface CreateSwarmParams {
  /** The agents in the swarm. Names must be unique. */
  agents: AnyAgent[];
  /** The agent that answers when a thread has no active agent yet. */
  defaultActiveAgent: string;
  /** Where to keep the active agent. Defaults to thread metadata in the default agent's memory. */
  store?: ActiveAgentStore;
  /** Most handoffs allowed in one turn before the swarm gives up. Defaults to 10. */
  maxHandoffs?: number;
}

export interface SwarmCallOptions {
  memory: ThreadRef;
  /** Start this turn with this agent instead of the thread's active agent. */
  activeAgent?: string;
  requestContext?: RequestContext;
}

export interface Handoff {
  from: string;
  to: string;
}

export interface SwarmResult {
  text: string;
  activeAgent: string;
  handoffs: Handoff[];
}

export type SwarmChunk =
  | ChunkType
  | { type: 'swarm-agent-start'; payload: { agentName: string } }
  | { type: 'swarm-handoff'; payload: Handoff };

export interface SwarmStreamOutput {
  fullStream: ReadableStream<SwarmChunk>;
  textStream: ReadableStream<string>;
  text: Promise<string>;
  activeAgent: Promise<string>;
  handoffs: Promise<Handoff[]>;
}

const ACTIVE_AGENT_KEY = 'activeAgent';

function threadMetadataStore(memory: MastraMemory): ActiveAgentStore {
  return {
    async get({ thread }) {
      const stored = await memory.getThreadById({ threadId: thread });
      const active = stored?.metadata?.[ACTIVE_AGENT_KEY];
      return typeof active === 'string' ? active : undefined;
    },
    async set({ thread, resource }, agentName) {
      const stored = await memory.getThreadById({ threadId: thread });
      if (!stored) {
        await memory.createThread({
          threadId: thread,
          resourceId: resource,
          metadata: { [ACTIVE_AGENT_KEY]: agentName },
        });
        return;
      }
      await memory.updateThread({
        id: thread,
        title: stored.title ?? '',
        metadata: { ...stored.metadata, [ACTIVE_AGENT_KEY]: agentName },
      });
    },
  };
}

// Before @mastra/core 1.76 (mastra-ai/mastra#26685), a run continuing a thread folded its first tool call into the
// previous assistant row unless that row carried this flag, so the call read as a parallel call of the agent that handed off.
async function endResponseAt(agent: AnyAgent, messageId: string, requestContext: RequestContext) {
  const memory = await agent.getMemory({ requestContext });
  const messages = await memory?.storage.getStore('memory');
  if (!messages) return;
  const [row] = (await messages.listMessagesById({ messageIds: [messageId] })).messages;
  if (!row) return;
  const metadata = (row.content.metadata ?? {}) as Record<string, unknown>;
  const mastra = (metadata.mastra ?? {}) as Record<string, unknown>;
  await messages.updateMessages({
    messages: [
      {
        id: messageId,
        content: { ...row.content, metadata: { ...metadata, mastra: { ...mastra, responseBoundary: true } } },
      },
    ],
  });
}

/** Creates a swarm: agents hand the conversation to each other, and each thread remembers who is active. */
export function createSwarm({
  agents,
  defaultActiveAgent,
  store,
  maxHandoffs = 10,
}: CreateSwarmParams) {
  if (agents.length === 0) throw new Error('A swarm needs at least one agent.');

  const byName = new Map<string, AnyAgent>();
  for (const agent of agents) {
    if (!agent.name) throw new Error('Every agent in a swarm needs a name.');
    if (byName.has(agent.name)) {
      throw new Error(`Agent with name '${agent.name}' already exists. Agent names must be unique.`);
    }
    byName.set(agent.name, agent);
  }
  if (!byName.has(defaultActiveAgent)) {
    throw new Error(
      `Default active agent '${defaultActiveAgent}' not found in agents ${[...byName.keys()].join(', ')}.`,
    );
  }

  const agentNamed = (name: string) => {
    const agent = byName.get(name);
    if (!agent) throw new Error(`Agent '${name}' not found in the swarm.`);
    return agent;
  };

  async function resolveStore(): Promise<ActiveAgentStore> {
    if (store) return store;
    const memory = await agentNamed(defaultActiveAgent).getMemory();
    if (!memory) {
      throw new Error(
        `The swarm needs memory to remember the active agent: give '${defaultActiveAgent}' a memory, or pass a store to createSwarm.`,
      );
    }
    return threadMetadataStore(memory);
  }

  async function validateHandoffs(requestContext?: RequestContext) {
    for (const [name, agent] of byName) {
      for (const destination of await getHandoffDestinations(agent, { requestContext })) {
        if (!byName.has(destination)) {
          throw new Error(`'${name}' hands off to '${destination}', which is not in the swarm.`);
        }
      }
    }
  }

  async function getActiveAgent(ref: ThreadRef): Promise<string> {
    return (await (await resolveStore()).get(ref)) ?? defaultActiveAgent;
  }

  async function setActiveAgent(ref: ThreadRef, agentName: string): Promise<void> {
    agentNamed(agentName);
    await (await resolveStore()).set(ref, agentName);
  }

  async function* run(
    messages: MessageInput,
    options: SwarmCallOptions,
    result: SwarmResult,
  ): AsyncGenerator<SwarmChunk> {
    await validateHandoffs(options.requestContext);
    const activeStore = await resolveStore();
    let current = options.activeAgent ?? (await activeStore.get(options.memory)) ?? defaultActiveAgent;
    agentNamed(current);
    if (options.activeAgent) await activeStore.set(options.memory, current);

    const requestContext = options.requestContext ?? new RequestContext();
    let input: MessageInput = messages;

    for (;;) {
      requestContext.delete(METADATA_KEY_HANDOFF_DESTINATION);
      yield { type: 'swarm-agent-start', payload: { agentName: current } };

      const output = await agentNamed(current).stream(input, {
        memory: options.memory,
        requestContext,
        stopWhen: () => requestContext.has(METADATA_KEY_HANDOFF_DESTINATION),
      });

      let text = '';
      for await (const chunk of output.fullStream as AsyncIterable<ChunkType>) {
        if (chunk.type === 'text-delta') text += chunk.payload.text;
        yield chunk;
      }

      const next = requestContext.get(METADATA_KEY_HANDOFF_DESTINATION) as string | undefined;
      requestContext.delete(METADATA_KEY_HANDOFF_DESTINATION);
      if (!next) {
        result.text = text;
        result.activeAgent = current;
        return;
      }

      if (result.handoffs.length >= maxHandoffs) {
        throw new Error(
          `Swarm exceeded maxHandoffs (${maxHandoffs}) in one turn: ${[...result.handoffs.map((h) => h.from), current, next].join(' -> ')}.`,
        );
      }
      agentNamed(next);
      await endResponseAt(agentNamed(current), output.messageId, requestContext);
      const handoff = { from: current, to: next };
      result.handoffs.push(handoff);
      yield { type: 'swarm-handoff', payload: handoff };
      await activeStore.set(options.memory, next);
      current = next;
      input = [];
    }
  }

  function stream(messages: MessageInput, options: SwarmCallOptions): SwarmStreamOutput {
    const result: SwarmResult = { text: '', activeAgent: '', handoffs: [] };
    let settle!: { resolve: (r: SwarmResult) => void; reject: (e: unknown) => void };
    const done = new Promise<SwarmResult>((resolve, reject) => (settle = { resolve, reject }));
    done.catch(() => undefined);

    const source = new ReadableStream<SwarmChunk>({
      start(controller) {
        void (async () => {
          try {
            for await (const chunk of run(messages, options, result)) controller.enqueue(chunk);
            controller.close();
            settle.resolve(result);
          } catch (error) {
            controller.error(error);
            settle.reject(error);
          }
        })();
      },
    });

    const [fullStream, forText] = source.tee();
    const textStream = forText.pipeThrough(
      new TransformStream<SwarmChunk, string>({
        transform(chunk, controller) {
          if (chunk.type === 'text-delta') controller.enqueue(chunk.payload.text);
        },
      }),
    );

    return {
      fullStream,
      textStream,
      get text() {
        return done.then((r) => r.text);
      },
      get activeAgent() {
        return done.then((r) => r.activeAgent);
      },
      get handoffs() {
        return done.then((r) => r.handoffs);
      },
    };
  }

  async function generate(messages: MessageInput, options: SwarmCallOptions): Promise<SwarmResult> {
    const result: SwarmResult = { text: '', activeAgent: '', handoffs: [] };
    for await (const _chunk of run(messages, options, result)) {
      void _chunk;
    }
    return result;
  }

  return { stream, generate, getActiveAgent, setActiveAgent, agents: byName };
}

export type Swarm = ReturnType<typeof createSwarm>;
