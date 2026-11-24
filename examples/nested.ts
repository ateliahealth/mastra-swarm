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

const handoffs = (...targets: Array<[string, string]>) =>
  Object.fromEntries(
    targets.map(([agentName, description]) => {
      const tool = createHandoffTool({ agentName, description });
      return [tool.id, tool];
    }),
  );

const toFrontDesk: [string, string] = ['frontdesk', 'Anything that is not a workflow: questions, greetings, other topics'];
const toWorkflows: [string, string] = ['workflows', 'Start or continue a workflow, such as a patient intake or a form'];
const toIntake: [string, string] = ['intake', 'Patient intake: collect a new patient name and reason for visit'];
const toForms: [string, string] = ['forms', 'Fill out or check the status of a form'];

const recordIntake = createTool({
  id: 'recordIntake',
  description: 'Record a patient intake',
  inputSchema: z.object({ patientName: z.string(), reason: z.string() }),
  execute: async ({ patientName }) => ({ intakeId: `IN-${patientName.length}42` }),
});

const frontDesk = new Agent({
  id: 'frontdesk',
  name: 'frontdesk',
  instructions:
    'You are the front desk of a clinic. For anything about workflows, call the workflows handoff tool right away, without telling the user first. Answer other questions briefly.',
  model,
  memory,
  tools: handoffs(toWorkflows),
});

const workflowManager = new Agent({
  id: 'workflow-manager',
  name: 'workflow-manager',
  instructions:
    'You start clinic workflows. Call the handoff tool for the workflow the user wants right away, without telling the user first. Never do the workflow yourself.',
  model,
  memory,
  tools: handoffs(toIntake, toForms, toFrontDesk),
});

const intake = new Agent({
  id: 'intake',
  name: 'intake',
  instructions:
    'You run patient intake. Ask for the patient name and reason for visit, then record it. For anything else, hand off.',
  model,
  memory,
  tools: { recordIntake, ...handoffs(toForms, toFrontDesk) },
});

const forms = new Agent({
  id: 'forms',
  name: 'forms',
  instructions: 'You help with clinic forms. For anything else, hand off.',
  model,
  memory,
  tools: handoffs(toIntake, toFrontDesk),
});

const workflows = createSwarm({
  name: 'workflows',
  agents: [workflowManager, intake, forms],
  defaultActiveAgent: 'workflow-manager',
});
const swarm = createSwarm({ agents: [frontDesk, workflows], defaultActiveAgent: 'frontdesk' });
const conversation = { memory: { thread: randomUUID(), resource: 'example-user' } };

for (const message of [
  'I need to start a patient intake',
  'Jane Doe, she has a sore throat',
  'what are your opening hours?',
  'I need another intake, for John Roe',
]) {
  console.log(`\nuser: ${message}`);
  const started = Date.now();
  const output = swarm.stream(message, conversation);
  for await (const chunk of output.fullStream) {
    if (chunk.type === 'swarm-agent-start') process.stdout.write(`${chunk.payload.path.join('/')}: `);
    if (chunk.type === 'swarm-handoff') process.stdout.write(`[→ ${chunk.payload.target ?? chunk.payload.to}]\n`);
    if (chunk.type === 'tool-call' && chunk.payload.toolName === 'recordIntake') process.stdout.write('[recordIntake] ');
    if (chunk.type === 'text-delta') process.stdout.write(chunk.payload.text);
  }
  console.log(`\n(active: ${(await output.activePath).join('/')}, ${Date.now() - started} ms)`);
}
