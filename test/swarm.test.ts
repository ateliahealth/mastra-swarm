import { describe, expect, it } from 'vitest';
import { Agent } from '@mastra/core/agent';
import { InMemoryStore } from '@mastra/core/storage';
import { createTool } from '@mastra/core/tools';
import { Memory } from '@mastra/memory';
import type { MockLanguageModelV2 } from 'ai/test';
import { z } from 'zod';
import { createHandoffTool, createSwarm, type ActiveAgentStore } from '../src';
import { scriptedModel, transcript, type ScriptedReply } from './scripted-model';

const BOB_GREETING =
  "Ahoy, matey! Bob the pirate be at yer service. What be ye needin' help with today on the high seas? Arrr!";
const ALICE_ANSWER = 'The sum of 5 and 7 is 12.';

const recordedReplies: ScriptedReply[] = [
  { toolCall: { name: 'transfer_to_bob', id: 'call_1LlFyjm6iIhDjdn7juWuPYr4' } },
  { text: BOB_GREETING },
  { toolCall: { name: 'transfer_to_alice', id: 'call_T6pNmo2jTfZEK3a9avQ14f8Q' } },
  { toolCall: { name: 'add', args: { a: 5, b: 7 }, id: 'call_4kLYO1amR2NfhAxfECkALCr1' } },
  { text: ALICE_ANSWER },
];

const add = createTool({
  id: 'add',
  description: 'Add two numbers.',
  inputSchema: z.object({ a: z.number(), b: z.number() }),
  execute: async ({ a, b }) => a + b,
});

const newMemory = () => new Memory({ storage: new InMemoryStore(), options: { lastMessages: 50 } });

function aliceAndBob(model: MockLanguageModelV2, memory: Memory | undefined = newMemory()) {
  const toBob = createHandoffTool({ agentName: 'Bob' });
  const toAlice = createHandoffTool({
    agentName: 'Alice',
    description: 'Transfer to Alice, she can help with math',
  });
  const alice = new Agent({
    id: 'alice',
    name: 'Alice',
    instructions: 'You are Alice, an addition expert.',
    model,
    memory,
    tools: { add, [toBob.id]: toBob },
  });
  const bob = new Agent({
    id: 'bob',
    name: 'Bob',
    instructions: 'You are Bob, you speak like a pirate.',
    model,
    memory,
    tools: { [toAlice.id]: toAlice },
  });
  return { alice, bob, memory };
}

const thread = (id = 'thread-1') => ({ memory: { thread: id, resource: 'user-1' } });

describe('Swarm (ported from langgraph-swarm)', () => {
  it('should run in basic case', async () => {
    const { model, calls } = scriptedModel(recordedReplies);
    const { alice, bob } = aliceAndBob(model);
    const swarm = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice' });

    const turn1 = await swarm.generate("i'd like to speak to Bob", thread());

    expect(turn1.text).toBe(BOB_GREETING);
    expect(turn1.activeAgent).toBe('Bob');
    expect(calls.map((c) => c.agent)).toEqual(['Alice', 'Bob']);
    expect(transcript(calls[1].prompt)).toEqual([
      "user: i'd like to speak to Bob",
      'assistant -> transfer_to_bob',
      'tool transfer_to_bob: Successfully transferred to Bob',
    ]);

    const turn2 = await swarm.generate("what's 5 + 7?", thread());

    expect(turn2.text).toBe(ALICE_ANSWER);
    expect(turn2.activeAgent).toBe('Alice');
    expect(calls.map((c) => c.agent)).toEqual(['Alice', 'Bob', 'Bob', 'Alice', 'Alice']);
    expect(transcript(calls[2].prompt)).toEqual([
      "user: i'd like to speak to Bob",
      'assistant -> transfer_to_bob',
      'tool transfer_to_bob: Successfully transferred to Bob',
      `assistant: ${BOB_GREETING}`,
      "user: what's 5 + 7?",
    ]);
    expect(transcript(calls[4].prompt)).toEqual([
      "user: i'd like to speak to Bob",
      'assistant -> transfer_to_bob',
      'tool transfer_to_bob: Successfully transferred to Bob',
      `assistant: ${BOB_GREETING}`,
      "user: what's 5 + 7?",
      'assistant -> transfer_to_alice',
      'tool transfer_to_alice: Successfully transferred to Alice',
      'assistant -> add',
      'tool add: 12',
    ]);
  });

  it('should run with a custom active agent store', async () => {
    const { model, calls } = scriptedModel(recordedReplies);
    const { alice, bob } = aliceAndBob(model);
    const saved = new Map<string, string>();
    const store: ActiveAgentStore = {
      get: async ({ thread }) => saved.get(thread),
      set: async ({ thread }, agentName) => void saved.set(thread, agentName),
    };
    const swarm = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice', store });

    const turn1 = await swarm.generate("i'd like to speak to Bob", thread());
    expect(turn1.activeAgent).toBe('Bob');
    expect(saved.get('thread-1')).toBe('Bob');

    const turn2 = await swarm.generate("what's 5 + 7?", thread());
    expect(turn2.text).toBe(ALICE_ANSWER);
    expect(saved.get('thread-1')).toBe('Alice');
    expect(calls.map((c) => c.agent)).toEqual(['Alice', 'Bob', 'Bob', 'Alice', 'Alice']);
  });

  it('should start from the active agent the caller provides', async () => {
    const { model, calls } = scriptedModel([{ text: BOB_GREETING }]);
    const { alice, bob } = aliceAndBob(model);
    const swarm = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice' });

    const turn = await swarm.generate('hello', { ...thread(), activeAgent: 'Bob' });

    expect(calls.map((c) => c.agent)).toEqual(['Bob']);
    expect(turn.activeAgent).toBe('Bob');
  });
});

describe('Swarm on Mastra', () => {
  it('sends a follow-up straight to the active agent without a router call', async () => {
    const { model, calls } = scriptedModel([
      ...recordedReplies,
      { toolCall: { name: 'add', args: { a: 10, b: 3 } } },
      { text: '13' },
    ]);
    const { alice, bob } = aliceAndBob(model);
    const swarm = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice' });

    await swarm.generate("i'd like to speak to Bob", thread());
    await swarm.generate("what's 5 + 7?", thread());
    const turn3 = await swarm.generate('and 10 + 3?', thread());

    expect(turn3.text).toBe('13');
    expect(calls.slice(5).map((c) => c.agent)).toEqual(['Alice', 'Alice']);
  });

  it('ends the handing-off agent at the handoff, even if it would keep going', async () => {
    const { model, calls } = scriptedModel((n) =>
      n === 0
        ? { toolCall: { name: 'transfer_to_bob' } }
        : n === 1
          ? { text: BOB_GREETING }
          : { text: 'Alice should never speak again' },
    );
    const { alice, bob } = aliceAndBob(model);
    const swarm = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice' });

    const turn = await swarm.generate("i'd like to speak to Bob", thread());

    expect(calls).toHaveLength(2);
    expect(turn.text).toBe(BOB_GREETING);
    expect(turn.handoffs).toEqual([{ from: 'Alice', to: 'Bob' }]);
  });

  it('keeps the active agent in thread metadata, across swarm instances', async () => {
    const { model, calls } = scriptedModel(recordedReplies);
    const { alice, bob, memory } = aliceAndBob(model);

    await createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice' }).generate(
      "i'd like to speak to Bob",
      thread(),
    );
    const stored = await memory!.getThreadById({ threadId: 'thread-1' });
    expect(stored?.metadata?.activeAgent).toBe('Bob');

    const fresh = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice' });
    expect(await fresh.getActiveAgent(thread().memory)).toBe('Bob');
    await fresh.generate("what's 5 + 7?", thread());
    expect(calls[2].agent).toBe('Bob');
  });

  it('keeps threads independent', async () => {
    const { model, calls } = scriptedModel([
      { toolCall: { name: 'transfer_to_bob' } },
      { text: BOB_GREETING },
      { text: 'Hi, I am Alice.' },
    ]);
    const { alice, bob } = aliceAndBob(model);
    const swarm = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice' });

    await swarm.generate("i'd like to speak to Bob", thread('a'));
    const other = await swarm.generate('hello', thread('b'));

    expect(other.activeAgent).toBe('Alice');
    expect(calls[2].agent).toBe('Alice');
  });

  it('lets the application move a thread to another agent', async () => {
    const { model, calls } = scriptedModel([{ text: BOB_GREETING }]);
    const { alice, bob } = aliceAndBob(model);
    const swarm = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice' });

    await swarm.setActiveAgent(thread().memory, 'Bob');
    await swarm.generate('hello', thread());

    expect(calls[0].agent).toBe('Bob');
  });

  it('stops a handoff loop at maxHandoffs', async () => {
    const { model } = scriptedModel((n) => ({
      toolCall: { name: n % 2 === 0 ? 'transfer_to_bob' : 'transfer_to_alice' },
    }));
    const { alice, bob } = aliceAndBob(model);
    const swarm = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice', maxHandoffs: 3 });

    await expect(swarm.generate('loop', thread())).rejects.toThrow(/maxHandoffs \(3\)/);
  });

  it('streams swarm events around each agent', async () => {
    const { model } = scriptedModel(recordedReplies);
    const { alice, bob } = aliceAndBob(model);
    const swarm = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice' });

    const output = swarm.stream("i'd like to speak to Bob", thread());
    const events: string[] = [];
    for await (const chunk of output.fullStream) {
      if (chunk.type === 'swarm-agent-start') events.push(`start ${chunk.payload.agentName}`);
      if (chunk.type === 'swarm-handoff') events.push(`${chunk.payload.from} -> ${chunk.payload.to}`);
    }

    expect(events).toEqual(['start Alice', 'Alice -> Bob', 'start Bob']);
    expect(await output.text).toBe(BOB_GREETING);
    expect(await output.activeAgent).toBe('Bob');
  });

  it('streams only text through textStream', async () => {
    const { model } = scriptedModel(recordedReplies);
    const { alice, bob } = aliceAndBob(model);
    const swarm = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice' });

    let text = '';
    for await (const delta of swarm.stream("i'd like to speak to Bob", thread()).textStream) text += delta;

    expect(text).toBe(BOB_GREETING);
  });
});

describe('createSwarm validation (ported from langgraph-swarm)', () => {
  const agents = () => aliceAndBob(scriptedModel([]).model);

  it('requires at least one agent', () => {
    expect(() => createSwarm({ agents: [], defaultActiveAgent: 'Alice' })).toThrow(/at least one agent/);
  });

  it('rejects duplicate agent names', () => {
    const { alice } = agents();
    expect(() => createSwarm({ agents: [alice, alice], defaultActiveAgent: 'Alice' })).toThrow(
      /Agent with name 'Alice' already exists/,
    );
  });

  it('rejects a default active agent that is not in the swarm', () => {
    const { alice, bob } = agents();
    expect(() => createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Carol' })).toThrow(
      /Default active agent 'Carol' not found/,
    );
  });

  it('rejects an active agent the caller provides that is not in the swarm', async () => {
    const { alice, bob } = agents();
    const swarm = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice' });
    await expect(swarm.generate('hi', { ...thread(), activeAgent: 'Carol' })).rejects.toThrow(
      /Agent 'Carol' not found/,
    );
  });

  it('rejects a handoff to an agent outside the swarm', async () => {
    const { model } = scriptedModel([]);
    const toCarol = createHandoffTool({ agentName: 'Carol' });
    const alice = new Agent({
      id: 'alice',
      name: 'Alice',
      instructions: 'You are Alice.',
      model,
      memory: newMemory(),
      tools: { [toCarol.id]: toCarol },
    });
    const swarm = createSwarm({ agents: [alice], defaultActiveAgent: 'Alice' });
    await expect(swarm.generate('hi', thread())).rejects.toThrow(
      /'Alice' hands off to 'Carol', which is not in the swarm/,
    );
  });

  it('requires memory or a store to remember the active agent', async () => {
    const { model } = scriptedModel([{ text: 'hi' }]);
    const alice = new Agent({ id: 'alice', name: 'Alice', instructions: 'You are Alice.', model });
    const swarm = createSwarm({ agents: [alice], defaultActiveAgent: 'Alice' });
    await expect(swarm.generate('hi', thread())).rejects.toThrow(/needs memory/);
  });
});

describe('handoff prompt', () => {
  const systemTexts = (prompt: { role: string; content: unknown }[]) =>
    prompt.filter((m) => m.role === 'system').map((m) => String(m.content));
  const last = (prompt: { role: string; content: unknown }[]) => prompt.at(-1);
  const mentionsHandoff = (prompt: { role: string; content: unknown }[]) =>
    systemTexts(prompt).some((t) => /handed to you/.test(t));

  it('ends the first call of the agent that receives a handoff with a reminder to answer the user', async () => {
    const { model, calls } = scriptedModel(recordedReplies);
    const { alice, bob, memory } = aliceAndBob(model);
    const swarm = createSwarm({ agents: [alice, bob], defaultActiveAgent: 'Alice' });

    await swarm.generate("i'd like to speak to Bob", thread());
    await swarm.generate("what's 5 + 7?", thread());

    expect(mentionsHandoff(calls[0].prompt)).toBe(false);
    expect(last(calls[1].prompt)).toMatchObject({ role: 'system' });
    expect(String(last(calls[1].prompt)?.content)).toMatch(/handed to you by Alice/);
    expect(mentionsHandoff(calls[2].prompt)).toBe(false);
    expect(String(last(calls[3].prompt)?.content)).toMatch(/handed to you by Bob/);
    expect(mentionsHandoff(calls[4].prompt)).toBe(false);

    const { messages } = await memory!.recall({ threadId: 'thread-1', resourceId: 'user-1' });
    expect(JSON.stringify(messages)).not.toMatch(/handed to you/);
  });

  it('can be replaced or turned off', async () => {
    const { model, calls } = scriptedModel(recordedReplies.slice(0, 2));
    const { alice, bob } = aliceAndBob(model);
    const custom = createSwarm({
      agents: [alice, bob],
      defaultActiveAgent: 'Alice',
      handoffPrompt: ({ from, to }) => `custom ${from} -> ${to}`,
    });
    await custom.generate("i'd like to speak to Bob", thread('custom'));
    expect(last(calls[1].prompt)).toMatchObject({ role: 'system', content: 'custom Alice -> Bob' });

    const off = scriptedModel(recordedReplies.slice(0, 2));
    const pair = aliceAndBob(off.model);
    const silent = createSwarm({ agents: [pair.alice, pair.bob], defaultActiveAgent: 'Alice', handoffPrompt: false });
    await silent.generate("i'd like to speak to Bob", thread('silent'));
    expect(mentionsHandoff(off.calls[1].prompt)).toBe(false);
  });
});
