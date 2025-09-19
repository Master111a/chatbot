import ChatDetailPage from '@/components/pages/chat/chat-detail-page';
import { use } from 'react';

export type PageProps = {
  params: Promise<{
    slug: string;
  }>;
};

export default function Page({ params }: PageProps) {
  const { slug } = use(params);
  return <ChatDetailPage slug={slug} />;
}
