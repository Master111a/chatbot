"use client";

import ChatITem from "@/components/Chat/side-bar/chat-item";
import { ButtonZy } from "@/components/ui/button-zy";
import appPath from "@/constants/app-path";
import { useApp } from "@/contexts/app-context";
import { useChat } from "@/contexts/chat-context";
import { useResponsive } from "@/hooks/use-responsive";
import { useTheme } from "@/contexts/theme-context";
import { cn } from "@/utils/cn";
import { Menu, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export const Sidebar = () => {
    const { chats } = useChat();
    const { isSidebarOpen, setIsSidebarOpen } = useApp();
    const router = useRouter();
    const { theme } = useTheme();
    const { isXs } = useResponsive();

    useEffect(() => {
        if (isXs) {
            setIsSidebarOpen(false);
        } else {
            setIsSidebarOpen(true);
        }
    }, [isXs, setIsSidebarOpen]);

    return (
        <>
            <div
                className={cn(
                    "fixed inset-y-0 left-0 z-30 transform transition-all duration-300 ease-in-out",
                    isSidebarOpen ? "translate-x-0" : "-translate-x-full"
                )}>
                <div
                    className="max-w-64 w-64 h-screen sidebar text-primary flex flex-col gap-y-5 theme-transition text-14"
                    style={{
                        background: `${
                            theme === "light"
                                ? "#ffffff"
                                : "color-mix(in oklab, var(--button-primary-bg) 90%, transparent)"
                        }`,
                    }}>
                    {/* New Chat Button - sidebar-top */}
                    <div className="flex items-center justify-between h-[3.5rem] px-3">
                        <div>
                            <ButtonZy
                                variant="text"
                                className="relative text-14 text-white h-[30px] w-fit px-3 py-1.5 text-nowrap bg-primary border-primary border-2 rounded-[4px] hover:bg-primary/85 font-600 uppercase leading-1 dark:text-primary dark:bg-[#002c5d] dark:hover:text-white dark:border-primary dark:hover:bg-[#002c5d] dark:hover:border-primary"
                                onClick={() => router.push(appPath.home)}>
                                New chat
                            </ButtonZy>
                        </div>
                        <div>
                            <ButtonZy
                                variant="text"
                                className="text-primary dark:text-primary hidden md:block"
                                onClick={() => setIsSidebarOpen(false)}>
                                <Menu className="size-[22px]" />
                            </ButtonZy>
                            <ButtonZy
                                variant="text"
                                className="text-primary dark:text-primary block md:hidden"
                                onClick={() => setIsSidebarOpen(false)}>
                                <X className="size-[22px]" />
                            </ButtonZy>
                        </div>
                    </div>
                    {/* Chat List */}
                    <div className="flex-1 px-3 overflow-y-auto w-full max-w-full">
                        {Boolean(chats.length) &&
                            chats?.map((chat) => (
                                <ChatITem key={chat.session_id} chat={chat} />
                            ))}
                    </div>
                </div>
            </div>
            {isSidebarOpen && (
                <div
                    className="fixed inset-0 bg-black/50 bg-opacity-50 z-20 md:hidden transition-opacity duration-300"
                    onClick={() => setIsSidebarOpen(false)}
                />
            )}
        </>
    );
};
