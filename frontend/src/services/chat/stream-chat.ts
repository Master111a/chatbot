export const streamChat = async ({
  message,
  sessionId,
  language = 'vi',
  onText,
  onEnd,
  onComplete,
}: {
  message: string;
  sessionId: string;
  language?: string;
  onText: (token: string) => void;
  onEnd?: () => void;
  onComplete?: (metadata: any) => void;
}) => {
  const response = await fetch(
    `${process.env.NEXT_PUBLIC_API_URL}/chat/stream`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        language,
      }),
    }
  );

  if (!response.body) throw new Error('No response body');

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');

    for (let i = 0; i < lines.length - 1; i++) {
      const line = lines[i].trim();
      if (!line.startsWith('data:')) continue;

      const jsonString = line.slice(5).trim(); 
      if (!jsonString) continue;

      try {
        const parsed = JSON.parse(jsonString);

        if (parsed.type === 'token') {
          const inner = JSON.parse(parsed.text);

          if (inner.type === 'response_token') {
            onText(inner.token);
          } else if (inner.type === 'stream_end') {
            onEnd?.();
          }
        } else if (parsed.type === 'stream_end') {
         
          onEnd?.();
        } else if (parsed.type === 'complete') {
          onComplete?.(parsed.metadata || {});
        } else if (parsed.type === 'response_complete') {
          if (parsed.text) {
            onText(parsed.text);
          }
          onEnd?.();
        } 
      } catch (err) {
        console.error('❌ Error parsing data line:', line, err);
      }
    }

    buffer = lines[lines.length - 1];
  }
};
