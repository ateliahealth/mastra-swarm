import { simulateReadableStream } from 'ai';
import { MockLanguageModelV2 } from 'ai/test';
import type { LanguageModelV2Prompt, LanguageModelV2StreamPart } from '@ai-sdk/provider';

export type ScriptedReply =
  | { text: string }
  | { toolCall: { name: string; args?: Record<string, unknown>; id?: string } };

export type RecordedCall = { agent: string; prompt: LanguageModelV2Prompt };

const usage = { inputTokens: 1, outputTokens: 1, totalTokens: 2 };

function partsFor(reply: ScriptedReply, n: number): LanguageModelV2StreamPart[] {
  const start: LanguageModelV2StreamPart[] = [{ type: 'stream-start', warnings: [] }];
  if ('text' in reply) {
    return [
      ...start,
      { type: 'text-start', id: `t${n}` },
      { type: 'text-delta', id: `t${n}`, delta: reply.text },
      { type: 'text-end', id: `t${n}` },
      { type: 'finish', finishReason: 'stop', usage },
    ];
  }
  return [
    ...start,
    {
      type: 'tool-call',
      toolCallId: reply.toolCall.id ?? `call_${n}`,
      toolName: reply.toolCall.name,
      input: JSON.stringify(reply.toolCall.args ?? {}),
    },
    { type: 'finish', finishReason: 'tool-calls', usage },
  ];
}

const agentOf = (prompt: LanguageModelV2Prompt) => {
  const system = prompt
    .filter((m) => m.role === 'system')
    .map((m) => m.content)
    .join('\n');
  return system.match(/You are (\w+)/)?.[1] ?? 'unknown';
};

export function scriptedModel(replies: ScriptedReply[] | ((call: number) => ScriptedReply)) {
  const calls: RecordedCall[] = [];
  const next = (n: number) => (typeof replies === 'function' ? replies(n) : replies[n]);
  const model = new MockLanguageModelV2({
    doStream: async ({ prompt }) => {
      const n = calls.length;
      calls.push({ agent: agentOf(prompt), prompt });
      const reply = next(n);
      if (!reply) throw new Error(`scriptedModel ran out of replies at call ${n}`);
      return { stream: simulateReadableStream({ chunks: partsFor(reply, n) }) };
    },
  });
  return { model, calls };
}

const resultText = (output: unknown): string => {
  const value = (output as { value?: unknown })?.value ?? output;
  if (typeof value === 'string') return value;
  if (value && typeof value === 'object' && 'message' in value) {
    return String((value as { message: unknown }).message);
  }
  return JSON.stringify(value);
};

export function transcript(prompt: LanguageModelV2Prompt): string[] {
  const lines: string[] = [];
  for (const message of prompt) {
    if (message.role === 'system') continue;
    if (message.role === 'user') {
      lines.push(
        `user: ${message.content.map((p) => (p.type === 'text' ? p.text : `[${p.type}]`)).join('')}`,
      );
      continue;
    }
    for (const part of message.content) {
      if (part.type === 'text' && part.text) lines.push(`assistant: ${part.text}`);
      if (part.type === 'tool-call') lines.push(`assistant -> ${part.toolName}`);
      if (part.type === 'tool-result') {
        lines.push(`tool ${part.toolName}: ${resultText(part.output)}`);
      }
    }
  }
  return lines;
}
