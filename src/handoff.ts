import type { Agent } from '@mastra/core/agent';
import type { RequestContext } from '@mastra/core/request-context';
import { createTool } from '@mastra/core/tools';
import { z } from 'zod';

export const METADATA_KEY_HANDOFF_DESTINATION = '__handoff_destination';

const WHITESPACE_RE = /\s+/g;
const destinations = new WeakMap<object, string>();

const normalizeAgentName = (agentName: string) =>
  agentName.trim().replace(WHITESPACE_RE, '_').toLowerCase();

export interface CreateHandoffToolParams {
  /** Name of the agent to hand the conversation to, as given to `new Agent({ name })`. */
  agentName: string;
  /** Tool name. Defaults to `transfer_to_<agent_name>`. */
  name?: string;
  /** Tool description. Defaults to `Ask agent '<agentName>' for help`. */
  description?: string;
}

/** Creates a tool that hands the conversation to another agent in the swarm. */
export function createHandoffTool({ agentName, name, description }: CreateHandoffToolParams) {
  const handoffTool = createTool({
    id: name ?? `transfer_to_${normalizeAgentName(agentName)}`,
    description: description ?? `Ask agent '${agentName}' for help`,
    inputSchema: z.object({}),
    execute: async (_input, context) => {
      context?.requestContext?.set(METADATA_KEY_HANDOFF_DESTINATION, agentName);
      return `Successfully transferred to ${agentName}`;
    },
  });
  destinations.set(handoffTool, agentName);
  return handoffTool;
}

/** The agent a handoff tool transfers to, or `undefined` for any other tool. */
export function getHandoffDestination(tool: unknown): string | undefined {
  return typeof tool === 'object' && tool !== null ? destinations.get(tool) : undefined;
}

/** The agents this agent can hand off to, read from its handoff tools. */
export async function getHandoffDestinations(
  agent: Agent<any, any, any>,
  options: { requestContext?: RequestContext } = {},
): Promise<string[]> {
  const tools = await agent.listTools(options);
  return Object.values(tools ?? {})
    .map(getHandoffDestination)
    .filter((destination): destination is string => destination !== undefined);
}
