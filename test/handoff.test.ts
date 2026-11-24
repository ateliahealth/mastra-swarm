import { describe, expect, it } from 'vitest';
import { Agent } from '@mastra/core/agent';
import { createTool } from '@mastra/core/tools';
import { z } from 'zod';
import * as pkg from '../src';
import { createHandoffTool, getHandoffDestination, getHandoffDestinations } from '../src';
import { scriptedModel } from './scripted-model';

describe('createHandoffTool (ported from langgraph-swarm)', () => {
  it('names the tool transfer_to_<agent>', () => {
    expect(createHandoffTool({ agentName: 'Bob' }).id).toBe('transfer_to_bob');
  });

  it('normalizes whitespace and case in the agent name', () => {
    expect(createHandoffTool({ agentName: '  Customer   Support ' }).id).toBe(
      'transfer_to_customer_support',
    );
  });

  it('accepts a custom tool name', () => {
    expect(createHandoffTool({ agentName: 'Bob', name: 'call_bob' }).id).toBe('call_bob');
  });

  it("describes itself as 'Ask agent <name> for help' by default", () => {
    expect(createHandoffTool({ agentName: 'Bob' }).description).toBe("Ask agent 'Bob' for help");
  });

  it('accepts a custom description', () => {
    const tool = createHandoffTool({ agentName: 'Alice', description: 'Transfer to Alice, she can help with math' });
    expect(tool.description).toBe('Transfer to Alice, she can help with math');
  });

  it('records its destination', () => {
    expect(getHandoffDestination(createHandoffTool({ agentName: 'Bob' }))).toBe('Bob');
    const add = createTool({ id: 'add', description: 'add', inputSchema: z.object({}), execute: async () => 0 });
    expect(getHandoffDestination(add)).toBeUndefined();
  });
});

describe('getHandoffDestinations (ported from langgraph-swarm)', () => {
  const { model } = scriptedModel([]);

  it("lists the agents an agent can hand off to", async () => {
    const toBob = createHandoffTool({ agentName: 'Bob' });
    const toCarol = createHandoffTool({ agentName: 'Carol' });
    const alice = new Agent({
      id: 'alice',
      name: 'Alice',
      instructions: 'You are Alice.',
      model,
      tools: { [toBob.id]: toBob, [toCarol.id]: toCarol },
    });
    expect(await getHandoffDestinations(alice)).toEqual(['Bob', 'Carol']);
  });

  it('returns nothing for an agent without handoff tools', async () => {
    const bob = new Agent({ id: 'bob', name: 'Bob', instructions: 'You are Bob.', model });
    expect(await getHandoffDestinations(bob)).toEqual([]);
  });
});

describe('package exports (ported from langgraph-swarm test_import)', () => {
  it('exposes the public API', () => {
    expect(Object.keys(pkg).sort()).toEqual(
      [
        'METADATA_KEY_HANDOFF_DESTINATION',
        'createHandoffTool',
        'createSwarm',
        'getHandoffDestination',
        'getHandoffDestinations',
      ].sort(),
    );
  });
});
