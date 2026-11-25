import type { Agent, AgentExecutionOptions } from '@mastra/core/agent';
import type { MastraMemory } from '@mastra/core/memory';
import { RequestContext } from '@mastra/core/request-context';
import type { ChunkType } from '@mastra/core/stream';
import { METADATA_KEY_HANDOFF_DESTINATION, getHandoffDestinations } from './handoff';

type AnyAgent = Agent<any, any, any>;
type MessageInput = Parameters<AnyAgent['stream']>[0];
type PrepareStep = NonNullable<AgentExecutionOptions['prepareStep']>;

export interface ThreadRef {
  thread: string;
  resource: string;
}

/**
 * Where the swarm remembers each thread's active member. Defaults to the thread's metadata.
 * `scope` is undefined for the top-level swarm and the nested swarm's name below it.
 */
export interface ActiveAgentStore {
  get(ref: ThreadRef, scope?: string): Promise<string | undefined>;
  set(ref: ThreadRef, agentName: string, scope?: string): Promise<void>;
}

export interface CreateSwarmParams {
  /** The agents and nested swarms in the swarm. Names must be unique across the whole tree. */
  agents: Array<AnyAgent | Swarm>;
  /** The member that answers when a thread has no active member yet: an agent or a nested swarm. */
  defaultActiveAgent: string;
  /** Required when this swarm is a member of another swarm. Other agents hand off to it by this name. */
  name?: string;
  /** Where to keep the active member. Defaults to thread metadata in the default agent's memory. */
  store?: ActiveAgentStore;
  /** Most handoffs allowed in one turn before the swarm gives up. Defaults to 10. */
  maxHandoffs?: number;
  /**
   * A system message for the agent that receives a handoff, sent with that call only and never saved.
   * Without one, a model often answers the transfer ("Transferred.") or nothing at all instead of the user.
   * `false` turns it off.
   */
  handoffPrompt?: ((handoff: Handoff) => string) | false;
}

export interface SwarmCallOptions {
  memory: ThreadRef;
  /** Start this turn with this agent or nested swarm instead of the thread's active agent. */
  activeAgent?: string;
  requestContext?: RequestContext;
}

export const defaultHandoffPrompt = ({ from }: Handoff) =>
  `The conversation was just handed to you by ${from}. Reply now to the user's most recent message, as if they had asked you directly. Do not mention the handoff.`;

export interface Handoff {
  from: string;
  to: string;
  /** The name the agent handed off to, when it was a nested swarm rather than `to` itself. */
  target?: string;
}

export interface SwarmResult {
  text: string;
  activeAgent: string;
  /** Names from the top-level swarm down to the active agent, e.g. `['workflows', 'Carl']`. */
  activePath: string[];
  handoffs: Handoff[];
}

export type SwarmChunk =
  | ChunkType
  | { type: 'swarm-agent-start'; payload: { agentName: string; path: string[] } }
  | { type: 'swarm-handoff'; payload: Handoff };

export interface SwarmStreamOutput {
  fullStream: ReadableStream<SwarmChunk>;
  textStream: ReadableStream<string>;
  text: Promise<string>;
  activeAgent: Promise<string>;
  activePath: Promise<string[]>;
  handoffs: Promise<Handoff[]>;
}

const TREE = Symbol.for('mastra-swarm.tree');

interface SwarmTree {
  name?: string;
  defaultActiveAgent: string;
  members: Map<string, AnyAgent | SwarmTree>;
}

export interface Swarm {
  readonly name?: string;
  readonly [TREE]: SwarmTree;
  /** Every agent in the swarm, nested swarms included, by name. */
  readonly agents: Map<string, AnyAgent>;
  stream(messages: MessageInput, options: SwarmCallOptions): SwarmStreamOutput;
  generate(messages: MessageInput, options: SwarmCallOptions): Promise<SwarmResult>;
  getActiveAgent(ref: ThreadRef): Promise<string>;
  getActivePath(ref: ThreadRef): Promise<string[]>;
  setActiveAgent(ref: ThreadRef, agentName: string): Promise<void>;
}

const isSwarm = (member: AnyAgent | Swarm): member is Swarm =>
  typeof member === 'object' && member !== null && TREE in member;

const isTree = (member: AnyAgent | SwarmTree): member is SwarmTree =>
  'members' in member && member.members instanceof Map;

const metadataKey = (scope?: string) => (scope ? `activeAgent:${scope}` : 'activeAgent');

function threadMetadataStore(memory: MastraMemory): ActiveAgentStore {
  return {
    async get({ thread }, scope) {
      const stored = await memory.getThreadById({ threadId: thread });
      const active = stored?.metadata?.[metadataKey(scope)];
      return typeof active === 'string' ? active : undefined;
    },
    async set({ thread, resource }, agentName, scope) {
      const stored = await memory.getThreadById({ threadId: thread });
      if (!stored) {
        await memory.createThread({
          threadId: thread,
          resourceId: resource,
          metadata: { [metadataKey(scope)]: agentName },
        });
        return;
      }
      await memory.updateThread({
        id: thread,
        title: stored.title ?? '',
        metadata: { ...stored.metadata, [metadataKey(scope)]: agentName },
      });
    },
  };
}

type ModelCall = { prompt: Array<{ role: string; content: unknown }> };
type ModelMethod = (options: ModelCall) => unknown;

/** The model, with `text` appended as the last system message of every call it makes. Nothing is stored. */
function withTrailingSystemMessage<T extends object>(model: T, text: string): T {
  return new Proxy(model, {
    get(target, property, receiver) {
      const value = Reflect.get(target, property, receiver);
      if (property !== 'doStream' && property !== 'doGenerate') {
        return typeof value === 'function' ? value.bind(target) : value;
      }
      return (options: ModelCall) =>
        (value as ModelMethod).call(target, {
          ...options,
          prompt: [...options.prompt, { role: 'system', content: text }],
        });
    },
  });
}

// Mastra places per-call `system` and `context` messages before the history, where the handoff's tool result
// still reads as the last word; the reminder only works as the final message of the first call.
const remindOnFirstStep =
  (text: string): PrepareStep =>
  ({ stepNumber, model }) =>
    stepNumber === 0 ? { model: withTrailingSystemMessage(model, text) } : undefined;

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

function buildTree({ agents, defaultActiveAgent, name }: CreateSwarmParams): SwarmTree {
  if (agents.length === 0) throw new Error('A swarm needs at least one agent.');

  const members = new Map<string, AnyAgent | SwarmTree>();
  const seen = new Set<string>();
  const claim = (memberName: string) => {
    if (seen.has(memberName)) {
      throw new Error(`Agent with name '${memberName}' already exists. Agent names must be unique.`);
    }
    seen.add(memberName);
  };

  for (const member of agents) {
    if (isSwarm(member)) {
      const tree = member[TREE];
      if (!tree.name) throw new Error('A swarm inside another swarm needs a name.');
      claim(tree.name);
      for (const nested of index(tree).keys()) claim(nested);
      members.set(tree.name, tree);
    } else {
      if (!member.name) throw new Error('Every agent in a swarm needs a name.');
      claim(member.name);
      members.set(member.name, member);
    }
  }

  if (!members.has(defaultActiveAgent)) {
    throw new Error(
      `Default active agent '${defaultActiveAgent}' not found in agents ${[...members.keys()].join(', ')}.`,
    );
  }
  return { name, defaultActiveAgent, members };
}

/** Every name in the tree, mapped to the member names leading to it from the tree's top. */
function index(tree: SwarmTree, prefix: string[] = [], into = new Map<string, string[]>()) {
  for (const [memberName, member] of tree.members) {
    const path = [...prefix, memberName];
    into.set(memberName, path);
    if (isTree(member)) index(member, path, into);
  }
  return into;
}

/** Creates a swarm: agents hand the conversation to each other, and each thread remembers who is active. */
export function createSwarm(params: CreateSwarmParams): Swarm {
  const { store, maxHandoffs = 10, handoffPrompt = defaultHandoffPrompt } = params;
  const root = buildTree(params);
  const paths = index(root);

  const nodeAt = (path: string[]) => {
    let node: AnyAgent | SwarmTree = root;
    for (const step of path) node = (node as SwarmTree).members.get(step)!;
    return node;
  };

  const pathTo = (name: string) => {
    const path = paths.get(name);
    if (!path) throw new Error(`Agent '${name}' not found in the swarm.`);
    return path;
  };

  const agents = new Map<string, AnyAgent>();
  for (const [name, path] of paths) {
    const node = nodeAt(path);
    if (!isTree(node)) agents.set(name, node);
  }

  const defaultAgentName = () => {
    let node: SwarmTree = root;
    for (;;) {
      const member = node.members.get(node.defaultActiveAgent)!;
      if (!isTree(member)) return node.defaultActiveAgent;
      node = member;
    }
  };

  async function resolveStore(): Promise<ActiveAgentStore> {
    if (store) return store;
    const defaultAgent = defaultAgentName();
    const memory = await agents.get(defaultAgent)!.getMemory();
    if (!memory) {
      throw new Error(
        `The swarm needs memory to remember the active agent: give '${defaultAgent}' a memory, or pass a store to createSwarm.`,
      );
    }
    return threadMetadataStore(memory);
  }

  async function validateHandoffs(requestContext?: RequestContext) {
    for (const [name, agent] of agents) {
      for (const destination of await getHandoffDestinations(agent, { requestContext })) {
        if (!paths.has(destination)) {
          throw new Error(`'${name}' hands off to '${destination}', which is not in the swarm.`);
        }
      }
    }
  }

  async function descend(activeStore: ActiveAgentStore, ref: ThreadRef, prefix: string[]): Promise<string[]> {
    const tree = nodeAt(prefix) as SwarmTree;
    const stored = await activeStore.get(ref, prefix.at(-1));
    const memberName = stored && tree.members.has(stored) ? stored : tree.defaultActiveAgent;
    const path = [...prefix, memberName];
    return isTree(tree.members.get(memberName)!) ? descend(activeStore, ref, path) : path;
  }

  async function moveTo(activeStore: ActiveAgentStore, ref: ThreadRef, name: string): Promise<string[]> {
    const path = pathTo(name);
    for (const [depth, memberName] of path.entries()) {
      await activeStore.set(ref, memberName, depth === 0 ? undefined : path[depth - 1]);
    }
    return isTree(nodeAt(path)) ? descend(activeStore, ref, path) : path;
  }

  async function getActivePath(ref: ThreadRef): Promise<string[]> {
    return descend(await resolveStore(), ref, []);
  }

  async function getActiveAgent(ref: ThreadRef): Promise<string> {
    return (await getActivePath(ref)).at(-1)!;
  }

  async function setActiveAgent(ref: ThreadRef, agentName: string): Promise<void> {
    pathTo(agentName);
    await moveTo(await resolveStore(), ref, agentName);
  }

  async function* run(
    messages: MessageInput,
    options: SwarmCallOptions,
    result: SwarmResult,
  ): AsyncGenerator<SwarmChunk> {
    await validateHandoffs(options.requestContext);
    const activeStore = await resolveStore();
    let path = options.activeAgent
      ? await moveTo(activeStore, options.memory, options.activeAgent)
      : await descend(activeStore, options.memory, []);

    const requestContext = options.requestContext ?? new RequestContext();
    let input: MessageInput = messages;
    let receivedHandoff: Handoff | undefined;

    for (;;) {
      const current = path.at(-1)!;
      const agent = agents.get(current)!;
      requestContext.delete(METADATA_KEY_HANDOFF_DESTINATION);
      yield { type: 'swarm-agent-start', payload: { agentName: current, path } };

      const output = await agent.stream(input, {
        memory: options.memory,
        requestContext,
        stopWhen: () => requestContext.has(METADATA_KEY_HANDOFF_DESTINATION),
        ...(receivedHandoff && handoffPrompt
          ? { prepareStep: remindOnFirstStep(handoffPrompt(receivedHandoff)) }
          : {}),
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
        result.activePath = path;
        return;
      }

      if (result.handoffs.length >= maxHandoffs) {
        throw new Error(
          `Swarm exceeded maxHandoffs (${maxHandoffs}) in one turn: ${[...result.handoffs.map((h) => h.from), current, next].join(' -> ')}.`,
        );
      }
      pathTo(next);
      await endResponseAt(agent, output.messageId, requestContext);
      path = await moveTo(activeStore, options.memory, next);
      const to = path.at(-1)!;
      const handoff: Handoff = next === to ? { from: current, to } : { from: current, to, target: next };
      result.handoffs.push(handoff);
      yield { type: 'swarm-handoff', payload: handoff };
      receivedHandoff = handoff;
      input = [];
    }
  }

  const emptyResult = (): SwarmResult => ({ text: '', activeAgent: '', activePath: [], handoffs: [] });

  function stream(messages: MessageInput, options: SwarmCallOptions): SwarmStreamOutput {
    const result = emptyResult();
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
      get activePath() {
        return done.then((r) => r.activePath);
      },
      get handoffs() {
        return done.then((r) => r.handoffs);
      },
    };
  }

  async function generate(messages: MessageInput, options: SwarmCallOptions): Promise<SwarmResult> {
    const result = emptyResult();
    for await (const _chunk of run(messages, options, result)) {
      void _chunk;
    }
    return result;
  }

  return {
    name: params.name,
    [TREE]: root,
    agents,
    stream,
    generate,
    getActiveAgent,
    getActivePath,
    setActiveAgent,
  };
}
