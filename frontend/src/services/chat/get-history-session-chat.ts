import axiosInstance from '@/libs/axios-instance';
import { ChatHistoryType } from '@/types';
import { useQuery } from '@tanstack/react-query';

export const fetchChatHistory = async (
  sessionId: string
): Promise<ChatHistoryType> => {
  const response = await axiosInstance.get<ChatHistoryType>(
    `/chat/session/${sessionId}`
  );
  return response.data;
};

export const useChatHistory = (sessionId: string) => {
  return useQuery({
    queryKey: ['chat-history', sessionId],
    queryFn: () => fetchChatHistory(sessionId),
    enabled: !!sessionId, // only fetch when sessionId is provided
  });
};
