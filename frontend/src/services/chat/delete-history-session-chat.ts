import axiosInstance from '@/libs/axios-instance';
import { useMutation, useQueryClient } from '@tanstack/react-query';

export const deleteChat = async (sessionId: string): Promise<void> => {
  await axiosInstance.delete(`/chat/session/${sessionId}`);
};

export const useDeleteChat = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (sessionId: string) => deleteChat(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chat-history'] });
    },
  });
};
