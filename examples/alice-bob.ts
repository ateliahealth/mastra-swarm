import { randomUUID } from 'node:crypto';
import { createAzure } from '@ai-sdk/azure';
import { openai } from '@ai-sdk/openai';
import { Agent } from '@mastra/core/agent';
import { InMemoryStore } from '@mastra/core/storage';
import { createTool } from '@mastra/core/tools';
import { Memory } from '@mastra/memory';
import { z } from 'zod';
import { createHandoffTool, createSwarm } from '../src';

const model = process.env.AZURE_OPENAI_API_KEY
  ? createAzure({
      apiKey: process.env.AZURE_OPENAI_API_KEY,
      resourceName: process.env.AZURE_OPENAI_RESOURCE_NAME,
      apiVersion: process.env.AZURE_OPENAI_API_VERSION,
      useDeploymentBasedUrls: true,
    }).chat(process.env.AZURE_OPENAI_DEPLOYMENT ?? 'gpt-4o')
  : openai(process.env.OPENAI_MODEL ?? 'gpt-4o');

const memory = new Memory({ storage: new InMemoryStore() });

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
const conversation = { memory: { thread: randomUUID(), resource: 'example-user' } };

for (const message of ["i'd like to speak to Bob", "what's 5 + 7?", 'and 10 + 3?']) {
  console.log(`\nuser: ${message}`);
  const started = Date.now();
  const output = swarm.stream(message, conversation);
  for await (const chunk of output.fullStream) {
    if (chunk.type === 'swarm-agent-start') process.stdout.write(`${chunk.payload.agentName}: `);
    if (chunk.type === 'swarm-handoff') process.stdout.write(`[→ ${chunk.payload.to}]\n`);
    if (chunk.type === 'tool-call' && chunk.payload.toolName === 'add') process.stdout.write('[add] ');
    if (chunk.type === 'text-delta') process.stdout.write(chunk.payload.text);
  }
  console.log(`\n(active: ${await output.activeAgent}, ${Date.now() - started} ms)`);
}
