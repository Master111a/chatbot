'use client';

import { AppProvider } from '@/contexts/app-context';
import { ChatProvider } from '@/contexts/chat-context';
import { ThemeProvider } from '@/contexts/theme-context';
import { queryClient } from '@/libs/query-client';
import { QueryClientProvider } from '@tanstack/react-query';
import { Slide, ToastContainer } from 'react-toastify';

interface ProviderProps {
  children: React.ReactNode;
}

const Provider = ({ children }: ProviderProps) => {
  return (
    <ThemeProvider>
      <AppProvider>
        <QueryClientProvider client={queryClient}>
          <ChatProvider>{children}</ChatProvider>
        </QueryClientProvider>
      </AppProvider>
      <ToastContainer
        position="top-center"
        autoClose={2000}
        hideProgressBar
        newestOnTop={false}
        closeOnClick={false}
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="light"
        transition={Slide}
      />
    </ThemeProvider>
  );
};

export default Provider;
