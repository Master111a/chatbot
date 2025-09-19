import axiosInstance from '@/libs/axios-instance';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'react-toastify';

export const updateTitleChat = async (
  sessionId: string,
  title: string
): Promise<void> => {
  await axiosInstance.put(`/chat/session/${sessionId}/title`, {
    title,
  });
};

export const useUpdateTitleChat = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ sessionId, title }: { sessionId: string; title: string }) =>
      updateTitleChat(sessionId, title),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chat-history'] });
    },
    onError: () => {
      toast.error('Thay đổi không thành công!');
    },
  });
};
