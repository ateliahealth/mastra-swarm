import { describe, expect, it } from 'vitest';
import { Agent } from '@mastra/core/agent';
import { InMemoryStore } from '@mastra/core/storage';
import { Memory } from '@mastra/memory';
import type { MockLanguageModelV2 } from 'ai/test';
import { createHandoffTool, createSwarm, type ActiveAgentStore } from '../src';
import { scriptedModel, type ScriptedReply } from './scripted-model';

const newMemory = () => new Memory({ storage: new InMemoryStore(), options: { lastMessages: 50 } });

const handoffs = (...agentNames: string[]) =>
  Object.fromEntries(
    agentNames.map((agentName) => {
      const tool = createHandoffTool({ agentName });
      return [tool.id, tool];
    }),
  );

function agent(name: string, model: MockLanguageModelV2, memory: Memory, targets: string[]) {
  return new Agent({
    id: name.toLowerCase(),
    name,
    instructions: `You are ${name}.`,
    model,
    memory,
    tools: handoffs(...targets),
  });
}

function hospital(replies: ScriptedReply[] | ((n: number) => ScriptedReply), store?: ActiveAgentStore) {
  const { model, calls } = scriptedModel(replies);
  const memory = newMemory();
  const alice = agent('Alice', model, memory, ['workflows', 'Carl']);
  const wendy = agent('Wendy', model, memory, ['Carl', 'Alice']);
  const carl = agent('Carl', model, memory, ['Wendy', 'Alice']);
  const workflows = createSwarm({ name: 'workflows', agents: [wendy, carl], defaultActiveAgent: 'Wendy' });
  const swarm = createSwarm({ agents: [alice, workflows], defaultActiveAgent: 'Alice', store });
  return { swarm, workflows, calls, memory, model, agents: { alice, wendy, carl } };
}

const thread = { memory: { thread: 'thread-1', resource: 'user-1' } };
const to = (name: string): ScriptedReply => ({ toolCall: { name: `transfer_to_${name.toLowerCase()}` } });
const say = (text: string): ScriptedReply => ({ text });

describe('nested swarms', () => {
  it('hands off into a nested swarm, which starts at its default agent', async () => {
    const { swarm, calls } = hospital([to('workflows'), say('Wendy here')]);

    const turn = await swarm.generate('start a workflow', thread);

    expect(calls.map((c) => c.agent)).toEqual(['Alice', 'Wendy']);
    expect(turn.text).toBe('Wendy here');
    expect(turn.activeAgent).toBe('Wendy');
    expect(turn.activePath).toEqual(['workflows', 'Wendy']);
    expect(turn.handoffs).toEqual([{ from: 'Alice', to: 'Wendy', target: 'workflows' }]);
  });

  it('sends a follow-up straight to the active agent inside the nested swarm', async () => {
    const { swarm, calls } = hospital([to('workflows'), to('Carl'), say('Carl here'), say('still Carl')]);

    await swarm.generate('start a workflow', thread);
    const followUp = await swarm.generate('and then?', thread);

    expect(calls.map((c) => c.agent)).toEqual(['Alice', 'Wendy', 'Carl', 'Carl']);
    expect(followUp.text).toBe('still Carl');
    expect(followUp.activePath).toEqual(['workflows', 'Carl']);
  });

  it('lets an agent inside a nested swarm hand back out to the parent', async () => {
    const { swarm, calls } = hospital([to('workflows'), say('Wendy here'), to('Alice'), say('Alice again'), say('still Alice')]);

    await swarm.generate('start a workflow', thread);
    const out = await swarm.generate('something else', thread);
    const next = await swarm.generate('and?', thread);

    expect(out.activeAgent).toBe('Alice');
    expect(out.activePath).toEqual(['Alice']);
    expect(next.text).toBe('still Alice');
    expect(calls.map((c) => c.agent)).toEqual(['Alice', 'Wendy', 'Wendy', 'Alice', 'Alice']);
  });

  it('resumes the last active agent when the conversation returns to a nested swarm', async () => {
    const { swarm, calls } = hospital([
      to('workflows'),
      to('Carl'),
      say('Carl here'),
      to('Alice'),
      say('Alice here'),
      to('workflows'),
      say('Carl again'),
    ]);

    await swarm.generate('start a workflow', thread);
    await swarm.generate('something else', thread);
    const back = await swarm.generate('back to the workflow', thread);

    expect(back.text).toBe('Carl again');
    expect(calls.at(-1)?.agent).toBe('Carl');
  });

  it('hands off to an agent inside a nested swarm by its own name', async () => {
    const { swarm, calls } = hospital([to('Carl'), say('Carl here'), say('still Carl')]);

    const turn = await swarm.generate('I need Carl', thread);
    await swarm.generate('and?', thread);

    expect(turn.activePath).toEqual(['workflows', 'Carl']);
    expect(calls.map((c) => c.agent)).toEqual(['Alice', 'Carl', 'Carl']);
  });

  it('remembers each level in thread metadata', async () => {
    const { swarm, memory } = hospital([to('workflows'), to('Carl'), say('Carl here')]);

    await swarm.generate('start a workflow', thread);

    const stored = await memory.getThreadById({ threadId: 'thread-1' });
    expect(stored?.metadata?.activeAgent).toBe('workflows');
    expect(stored?.metadata?.['activeAgent:workflows']).toBe('Carl');
  });

  it('passes the nested swarm name to a custom store as the scope', async () => {
    const saved = new Map<string, string>();
    const key = (thread: string, scope?: string) => `${thread}|${scope ?? ''}`;
    const store: ActiveAgentStore = {
      get: async ({ thread }, scope) => saved.get(key(thread, scope)),
      set: async ({ thread }, agentName, scope) => void saved.set(key(thread, scope), agentName),
    };
    const { swarm } = hospital([to('workflows'), to('Carl'), say('Carl here')], store);

    await swarm.generate('start a workflow', thread);

    expect(Object.fromEntries(saved)).toEqual({ 'thread-1|': 'workflows', 'thread-1|workflows': 'Carl' });
  });

  it('reads and moves the active agent across levels', async () => {
    const { swarm, calls } = hospital([say('Carl here')]);

    expect(await swarm.getActiveAgent(thread.memory)).toBe('Alice');
    await swarm.setActiveAgent(thread.memory, 'Carl');
    expect(await swarm.getActiveAgent(thread.memory)).toBe('Carl');
    expect(await swarm.getActivePath(thread.memory)).toEqual(['workflows', 'Carl']);

    await swarm.generate('hi', thread);
    expect(calls[0].agent).toBe('Carl');
  });

  it('starts a turn inside a nested swarm when the caller names that agent', async () => {
    const { swarm, calls } = hospital([say('Carl here')]);

    const turn = await swarm.generate('hi', { ...thread, activeAgent: 'Carl' });

    expect(calls[0].agent).toBe('Carl');
    expect(turn.activePath).toEqual(['workflows', 'Carl']);
  });

  it('counts handoffs at every level toward maxHandoffs', async () => {
    const { model } = scriptedModel((n) => (n % 2 === 0 ? to('Carl') : to('Alice')));
    const memory = newMemory();
    const alice = agent('Alice', model, memory, ['Carl']);
    const carl = agent('Carl', model, memory, ['Alice']);
    const workflows = createSwarm({ name: 'workflows', agents: [carl], defaultActiveAgent: 'Carl' });
    const swarm = createSwarm({ agents: [alice, workflows], defaultActiveAgent: 'Alice', maxHandoffs: 3 });

    await expect(swarm.generate('loop', thread)).rejects.toThrow(/maxHandoffs \(3\)/);
  });

  it('streams the path of each agent that starts', async () => {
    const { swarm } = hospital([to('workflows'), to('Carl'), say('Carl here')]);

    const starts: string[] = [];
    for await (const chunk of swarm.stream('start a workflow', thread).fullStream) {
      if (chunk.type === 'swarm-agent-start') starts.push(chunk.payload.path.join('/'));
    }

    expect(starts).toEqual(['Alice', 'workflows/Wendy', 'workflows/Carl']);
  });

  it('still works on its own', async () => {
    const { model, calls } = scriptedModel([to('Carl'), say('Carl here')]);
    const memory = newMemory();
    const workflows = createSwarm({
      name: 'workflows',
      agents: [agent('Wendy', model, memory, ['Carl']), agent('Carl', model, memory, ['Wendy'])],
      defaultActiveAgent: 'Wendy',
    });

    const turn = await workflows.generate('hi', thread);

    expect(calls.map((c) => c.agent)).toEqual(['Wendy', 'Carl']);
    expect(turn.activePath).toEqual(['Carl']);
  });
});

describe('nested swarm validation', () => {
  const parts = () => {
    const { model } = scriptedModel([]);
    const memory = newMemory();
    return {
      model,
      memory,
      alice: agent('Alice', model, memory, []),
      wendy: agent('Wendy', model, memory, []),
    };
  };

  it('needs a name to be a member of another swarm', () => {
    const { alice, wendy } = parts();
    const unnamed = createSwarm({ agents: [wendy], defaultActiveAgent: 'Wendy' });
    expect(() => createSwarm({ agents: [alice, unnamed], defaultActiveAgent: 'Alice' })).toThrow(
      /A swarm inside another swarm needs a name/,
    );
  });

  it('rejects an agent name used at two levels', () => {
    const { model, memory, alice, wendy } = parts();
    const otherAlice = agent('Alice', model, memory, []);
    const workflows = createSwarm({ name: 'workflows', agents: [wendy, otherAlice], defaultActiveAgent: 'Wendy' });
    expect(() => createSwarm({ agents: [alice, workflows], defaultActiveAgent: 'Alice' })).toThrow(
      /Agent with name 'Alice' already exists/,
    );
  });

  it('rejects a swarm named like an agent', () => {
    const { alice, wendy } = parts();
    const named = createSwarm({ name: 'Alice', agents: [wendy], defaultActiveAgent: 'Wendy' });
    expect(() => createSwarm({ agents: [alice, named], defaultActiveAgent: 'Wendy' })).toThrow(
      /Agent with name 'Alice' already exists/,
    );
  });

  it('accepts a nested swarm as the default active member', async () => {
    const { model, calls } = scriptedModel([say('Wendy here')]);
    const memory = newMemory();
    const workflows = createSwarm({
      name: 'workflows',
      agents: [agent('Wendy', model, memory, [])],
      defaultActiveAgent: 'Wendy',
    });
    const swarm = createSwarm({ agents: [agent('Alice', model, memory, []), workflows], defaultActiveAgent: 'workflows' });

    const turn = await swarm.generate('hi', thread);

    expect(calls[0].agent).toBe('Wendy');
    expect(turn.activePath).toEqual(['workflows', 'Wendy']);
  });

  it('rejects a handoff to a name that is nowhere in the tree', async () => {
    const { model } = scriptedModel([]);
    const memory = newMemory();
    const wendy = agent('Wendy', model, memory, ['Zed']);
    const workflows = createSwarm({ name: 'workflows', agents: [wendy], defaultActiveAgent: 'Wendy' });
    const swarm = createSwarm({ agents: [agent('Alice', model, memory, []), workflows], defaultActiveAgent: 'Alice' });

    await expect(swarm.generate('hi', thread)).rejects.toThrow(/'Wendy' hands off to 'Zed', which is not in the swarm/);
  });

  it('accepts a handoff out of a nested swarm to an agent in its parent', async () => {
    const { model, calls } = scriptedModel([say('Wendy here')]);
    const memory = newMemory();
    const wendy = agent('Wendy', model, memory, ['Alice']);
    const workflows = createSwarm({ name: 'workflows', agents: [wendy], defaultActiveAgent: 'Wendy' });
    const swarm = createSwarm({ agents: [agent('Alice', model, memory, []), workflows], defaultActiveAgent: 'workflows' });

    await swarm.generate('hi', thread);

    expect(calls[0].agent).toBe('Wendy');
  });
});
