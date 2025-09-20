"use client";

import { Anchor, Loader2, Search } from "lucide-react";
import { RefObject, useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import ReactMarkdown from "react-markdown";

import ChatInput from "@/components/pages/chat/chat-input";
import { ButtonZy } from "@/components/ui/button-zy";
import { useChat } from "@/contexts/chat-context";
import { useScrollButtonVisibility } from "@/hooks/use-scroll-button-visibility";
import { useChatHistory } from "@/services/chat/get-history-session-chat";
import { streamChat } from "@/services/chat/stream-chat";
import { MessageType } from "@/types";
import appConstant from "@/constants/app";
import { Message } from "@/components/Chat/Message";
import AnimationLoadingMessage from "./animation-loading-message";
import { cn } from "@/utils/cn";

interface FormInputs {
    message: string;
}

const ChatDetailPage = ({ slug }: { slug: string }) => {
    const { newMessage, setNewMessage } = useChat();

    const [messages, setMessages] = useState<MessageType[]>([]);

    const [chatResponse, setChatResponse] = useState("");
    const [loadingResponseMessage, setLoadingResponseMessage] = useState(false);

    const userMessageRef = useRef<HTMLDivElement>(null);
    const messagesContainerRef = useRef<HTMLDivElement>(null);
    const endOfMessagesRef = useRef<HTMLDivElement>(null);

    const { control, handleSubmit, setValue, setFocus } = useForm<FormInputs>({
        defaultValues: { message: "" },
    });

    const { data, isLoading: isHistoryLoading } = useChatHistory(slug);

    useEffect(() => {
        if (data?.messages && data.messages.length > 0) {
            setMessages(data.messages);
        }
    }, [data]);

    useEffect(() => {
        setFocus("message");
    }, [setFocus]);

    useEffect(() => {
        endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    useEffect(() => {
        userMessageRef.current?.scrollIntoView({ block: "start" });
    }, []);

    const { showScrollButton } = useScrollButtonVisibility({
        messagesContainerRef: messagesContainerRef as RefObject<HTMLElement>,
        messages,
        chatResponse,
    });

    const scrollToBottom = () => {
        messagesContainerRef.current?.scrollTo({
            top: messagesContainerRef.current.scrollHeight,
            behavior: "smooth",
        });
    };

    const handleSend = async (message: string) => {
        setLoadingResponseMessage(true);
        setChatResponse("");

        const newMessage: MessageType = {
            message_id: Date.now().toString(),
            message,
            response: "",
            metadata: {
                follow_up_questions: [],
            },
        };

        setMessages((prev) => [...prev, newMessage]);

        let streamedResponse = "";

        await streamChat({
            message,
            sessionId: slug,
            onText: (token) => {
                streamedResponse += token;
                setChatResponse(streamedResponse);
            },
            onEnd: () => {
                setMessages((prev) => {
                    const updatedMessages = [...prev];
                    const lastMessage = {
                        ...updatedMessages[updatedMessages.length - 1],
                    };
                    if (lastMessage) {
                        lastMessage.response = streamedResponse;
                        updatedMessages[updatedMessages.length - 1] =
                            lastMessage;
                    }
                    return updatedMessages;
                });

                setChatResponse("");
                setLoadingResponseMessage(false);
            },
        });
    };

    const onSubmit = async (data: FormInputs) => {
        const message = data.message.trim();
        if (!message || loadingResponseMessage) return;

        setValue("message", "");
        try {
            await handleSend(message);
        } catch (error) {
            console.error("❌ Error sending message:", error);
            setLoadingResponseMessage(false);
        }
    };

    useEffect(() => {
        const handleSendMessage = async () => {
            if (newMessage) {
                await onSubmit({ message: newMessage });
                setNewMessage("");
            }
        };
        handleSendMessage();
    }, []);

    return (
        <div className="flex flex-col h-[calc(100vh-56px)]">
            <div
                ref={messagesContainerRef}
                className="flex-1 overflow-y-auto relative">
                {isHistoryLoading ? (
                    <div className="text-center mt-10 text-sm text-gray-500">
                        Loading conversation history...
                    </div>
                ) : (
                    <>
                        {messages.map((message, index) => (
                            <div
                                key={message.message_id}
                                ref={
                                    index === messages.length - 1
                                        ? userMessageRef
                                        : null
                                }
                                className="theme-transition">
                                <div className="max-w-3xl mx-auto py-6 px-4">
                                    <Message content={message} />
                                </div>
                            </div>
                        ))}

                        {loadingResponseMessage && chatResponse && (
                            <div className="animate-fadeIn max-w-3xl mx-auto mt-4 px-4">
                                <div className="prose dark:prose-invert w-full max-w-3xl">
                                    <ReactMarkdown>
                                        {chatResponse}
                                    </ReactMarkdown>
                                </div>
                            </div>
                        )}

                        {loadingResponseMessage && !chatResponse && (
                            <AnimationLoadingMessage
                                time={2000}
                                loadingText="Searching, please wait..."
                                loading={loadingResponseMessage}
                            />
                        )}
                    </>
                )}
                <div ref={endOfMessagesRef} />
            </div>

            <div className="relative mb-10 bg-[var(--chat-bg)] theme-transition rounded-t-8">
                <div className="max-w-3xl min-w-[320px] lg:min-w-[640px] mx-auto py-2 md:py-4 shadow-md rounded-28 border border-gray-200 dark:bg-[var(--button-primary-bg)] dark:border-transparent">
                    <form
                        onSubmit={handleSubmit(onSubmit)}
                        className="relative">
                        <ChatInput
                            name="message"
                            control={control}
                            onEnterPress={handleSubmit(onSubmit)}
                        />
                        <ButtonZy
                            disabled={
                                loadingResponseMessage || isHistoryLoading
                            }
                            type="submit"
                            className="absolute size-[36px] overflow-hidden right-4 bottom-[30px] translate-y-1/2 p-2 rounded-full dark:text-white">
                            {loadingResponseMessage ? (
                                <Loader2
                                    size={20}
                                    className="text-white animate-spin"
                                />
                            ) : (
                                <Search size={20} />
                            )}
                        </ButtonZy>
                    </form>
                </div>
                <div className="text-center text-xs text-[var(--text-secondary)] mt-2">
                    {appConstant.title} can make mistakes. Consider checking
                    important information.
                </div>

                <ButtonZy
                    onClick={scrollToBottom}
                    className={cn(
                        "absolute -top-[45px] right-1/2 translate-x-1/2 p-2 rounded-full shadow-lg bg-primary dark:text-white transition-all duration-200",
                        showScrollButton && "opacity-100 visible",
                        !showScrollButton && "opacity-0 invisible"
                    )}>
                    <Anchor size={16} />
                </ButtonZy>
            </div>
        </div>
    );
};

export default ChatDetailPage;
