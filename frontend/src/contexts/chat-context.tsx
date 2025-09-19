"use client";

import appPath from "@/constants/app-path";
import { useDeleteChat } from "@/services/chat/delete-history-session-chat";
import { useUpdateTitleChat } from "@/services/chat/update-title-chat";
import { ChatHistoryType, SessionType } from "@/types";
import { useParams, useRouter } from "next/navigation";
import React, {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useState,
} from "react";
import { toast } from "react-toastify";

export interface Chat {
    session_id: string;
    title: string;
}

interface ChatContextType {
    chats: Chat[];
    setChats: (chats: Chat[]) => void;
    currentChat: ChatHistoryType | null;
    setCurrentChat: (currentChat: ChatHistoryType | null) => void;
    createNewChat: (session: SessionType) => Chat;
    selectChat: (chatId: string) => void;
    updateChatTitle: (chatId: string, title: string) => void;
    deleteChat: (chatId: string) => void;
    newMessage: string;
    setNewMessage: (newMessage: string) => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const ChatProvider: React.FC<{ children: React.ReactNode }> = ({
    children,
}) => {
    const [chats, setChats] = useState<Chat[]>([]);
    const [currentChat, setCurrentChat] = useState<ChatHistoryType | null>(
        null
    );
    const [newMessage, setNewMessage] = useState<string>("");

    const { mutate: deleteChatMutation } = useDeleteChat();

    const router = useRouter();
    const { slug } = useParams();

    // get data from localStorage
    useEffect(() => {
        const storedChats = localStorage.getItem("chat-session");
        if (storedChats) {
            setChats(JSON.parse(storedChats));
            if (slug) {
                const foundChat = JSON.parse(storedChats).find(
                    (chat: Chat) => chat.session_id === slug
                );
                if (foundChat) {
                    setCurrentChat(foundChat);
                }
            }
        }
    }, [slug]);

    // write to localStorage when chats change
    useEffect(() => {
        localStorage.setItem("chat-session", JSON.stringify(chats));
    }, [chats]);

    // listen to change from other tab to sync
    useEffect(() => {
        const handleStorageChange = (event: StorageEvent) => {
            if (event.key === "chat-session" && event.newValue) {
                setChats(JSON.parse(event.newValue));
            }
        };

        window.addEventListener("storage", handleStorageChange);
        return () => {
            window.removeEventListener("storage", handleStorageChange);
        };
    }, []);

    const createNewChat = useCallback((session: SessionType): Chat => {
        const newChat: Chat = {
            session_id: session.session_id,
            title: session.title,
        };
        setChats((prev) => [newChat, ...prev]);
        setCurrentChat({
            session_id: session.session_id,
            title: session.title,
            messages: [],
        });
        return newChat;
    }, []);

    const selectChat = useCallback(
        (chatId: string) => {
            const chat = chats.find((c) => c.session_id === chatId);
            if (chat) {
                setCurrentChat(chat as ChatHistoryType);
            }
        },
        [chats]
    );

    const { mutate: updateTitleChatMutation } = useUpdateTitleChat();

    const updateChatTitle = useCallback(
        (chatId: string, title: string) => {
            const prevChats = chats;
            const prevCurrentChat = currentChat;
            setChats((prev) =>
                prev.map((chat) =>
                    chat.session_id === chatId ? { ...chat, title } : chat
                )
            );
            setCurrentChat((prev) =>
                prev?.session_id === chatId ? { ...prev, title } : prev
            );

            updateTitleChatMutation(
                { sessionId: chatId, title },
                {
                    onError: () => {
                        setChats(prevChats);
                        setCurrentChat(prevCurrentChat);
                        toast("Update chat title failed!");
                    },
                }
            );
        },
        [chats, currentChat, updateTitleChatMutation]
    );

    const deleteChat = useCallback(
        (chatId: string) => {
            deleteChatMutation(chatId, {
                onSuccess: () => {
                    if (currentChat?.session_id === chatId) {
                        setCurrentChat(null);
                        router.push(appPath.home);
                    }
                    setChats((prev) =>
                        prev.filter((chat) => chat.session_id !== chatId)
                    );
                    // update localStorage -> handle by useEffect(chats)
                },
                onError: () => {
                    toast("Delete chat failed!");
                },
            });
        },
        [currentChat?.session_id, deleteChatMutation, router]
    );

    return (
        <ChatContext.Provider
            value={{
                chats,
                setChats,
                currentChat,
                setCurrentChat,
                createNewChat,
                selectChat,
                updateChatTitle,
                deleteChat,
                newMessage,
                setNewMessage,
            }}>
            {children}
        </ChatContext.Provider>
    );
};

export const useChat = () => {
    const context = useContext(ChatContext);
    if (!context) {
        throw new Error("useChat must be used within a ChatProvider");
    }
    return context;
};
