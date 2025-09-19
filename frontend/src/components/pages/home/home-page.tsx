"use client";

import ChatInput from "@/components/pages/chat/chat-input";
import { ButtonZy } from "@/components/ui/button-zy";
import appPath from "@/constants/app-path";
import { useChat } from "@/contexts/chat-context";
import { useCreateSession } from "@/services/chat/create-session-chat";
import { Loader2, Ship } from "lucide-react";
import { GiWaveSurfer } from "react-icons/gi";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";

const HomePage = () => {
    const [isSubmitting, setIsSubmitting] = useState(false);
    const { control, handleSubmit, setFocus, reset } = useForm({
        defaultValues: {
            message: "",
        },
    });

    const { createNewChat, setNewMessage } = useChat();
    const router = useRouter();
    const { mutateAsync: createSession } = useCreateSession();

    const onSubmit = async (data: { message: string }) => {
        if (!data.message.trim() || isSubmitting) return;

        try {
            setIsSubmitting(true);
            const session = await createSession({
                language: "vi",
                query: data.message,
            });

            const payload = {
                session_id: session.session_id,
                title: session.title,
            };

            const newChat = createNewChat(payload);
            setNewMessage(data.message);
            router.push(appPath.chatDetail(newChat.session_id));
        } finally {
            setIsSubmitting(false);
            reset();
        }
    };

    useEffect(() => {
        setFocus("message");
    }, [setFocus]);

    return (
        <div className="flex flex-col items-center justify-center h-[calc(100vh-56px)] gap-y-6 pr-[15px]">
            <div className="flex flex-col w-full items-center justify-center gap-y-6 -translate-y-1/2">
                <h1 className="flex items-end gap-2">
                    <GiWaveSurfer className="wave-icon text-white dark:text-primary size-8 sm:size-10 cs-animation-spin i" />
                    <span className="text-18 sm:text-24 font-700 text-white dark:text-primary">
                        Hi Newer! What can I help with?
                    </span>
                </h1>
                <div className="max-w-3xl w-full lg:min-w-[640px] mx-auto mt-auto p-2 md:p-4 shadow-md rounded-28 bg-white border border-gray-200 dark:bg-[var(--button-primary-bg)]/90 dark:border-transparent">
                    <form
                        onSubmit={handleSubmit(onSubmit)}
                        className="relative">
                        <ChatInput
                            name="message"
                            control={control}
                            onEnterPress={handleSubmit(onSubmit)}
                        />
                        <ButtonZy
                            type="submit"
                            disabled={isSubmitting}
                            className="absolute cursor-pointer right-2 top-1/2 -translate-y-1/2 p-2 rounded-full transition-all duration-200 theme-transition">
                            {isSubmitting ? (
                                <Loader2
                                    size={20}
                                    className="text-white animate-spin"
                                />
                            ) : (
                                <Ship size={20} className="text-white" />
                            )}
                        </ButtonZy>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default HomePage;
