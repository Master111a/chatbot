// hooks/useCreateSession.ts
import axiosInstance from '@/libs/axios-instance';
import { useMutation } from '@tanstack/react-query';

interface CreateSessionInput {
  language: string;
  query: string;
}

interface CreateSessionResponse {
  session_id: string;
  title: string;
}

export const useCreateSession = () =>
  useMutation<CreateSessionResponse, Error, CreateSessionInput>({
    mutationFn: async (data) => {
      const res = await axiosInstance.post('/chat/create-session', data);
      return res.data;
    },
  });
