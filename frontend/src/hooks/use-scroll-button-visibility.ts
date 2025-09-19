import { RefObject, useEffect, useState } from 'react';

interface UseScrollButtonVisibilityProps {
  messagesContainerRef: RefObject<HTMLElement>;
  messages: any[];
  chatResponse: string;
}

export const useScrollButtonVisibility = ({
  messagesContainerRef,
  messages,
  chatResponse,
}: UseScrollButtonVisibilityProps) => {
  const [showScrollButton, setShowScrollButton] = useState(false);

  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const checkScrollable = () => {
      const hasScrollableContent =
        container.scrollHeight > container.clientHeight;
      const isAtBottom =
        Math.abs(
          container.scrollHeight - container.scrollTop - container.clientHeight
        ) < 20;

      setShowScrollButton(hasScrollableContent && !isAtBottom);
    };

    const observer = new MutationObserver(checkScrollable);
    observer.observe(container, {
      childList: true,
      subtree: true,
      characterData: true,
    });

    container.addEventListener('scroll', checkScrollable);
    checkScrollable();

    return () => {
      container.removeEventListener('scroll', checkScrollable);
      observer.disconnect();
    };
  }, [messages, chatResponse, messagesContainerRef]);

  return { showScrollButton };
};
