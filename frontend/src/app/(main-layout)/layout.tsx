import MainChatArea from '@/components/Chat/main-chat-area';
import { Sidebar } from '@/components/Chat/side-bar/side-bar';
import appConstant from '@/constants/app';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: appConstant.title,
  description: 'AI by Newwave Solutions',
};

export default function MainLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <div>
      <Sidebar />
      {/* Main chat area */}
      <MainChatArea>{children}</MainChatArea>
    </div>
  );
}
