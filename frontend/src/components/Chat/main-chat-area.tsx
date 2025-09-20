"use client";

import { useApp } from "@/contexts/app-context";
import { cn } from "@/utils/cn";
import { Topbar } from "./top-bar/top-bar";
import { useTheme } from "@/contexts/theme-context";

interface MainChatAreaProps {
    children: React.ReactNode;
}

const MainChatArea = ({ children }: MainChatAreaProps) => {
    const { isSidebarOpen } = useApp();
    const { theme } = useTheme();
    return (
        <div
            className={cn(
                "flex-1 flex flex-col min-h-screen transition-all duration-500 ease-in-out bg-center bg-cover grow-1 bg-no-repeat relative",
                isSidebarOpen ? "md:ml-64" : "md:ml-0"
            )}
            style={{
                backgroundImage: `${
                    theme === "light"
                        ? "url('/BGligth.png')"
                        : "url('/dark.jpg')"
                }`,
            }}>
            <div className="relative z-[2]">
                <Topbar />
                <div className="w-full">{children}</div>
            </div>
            {theme !== "light" && (
                <div className="bg-overlay absolute top-0 bottom-0 left-0 right-0 bg-[var(--button-primary-bg)]/90"></div>
            )}
        </div>
    );
};

export default MainChatArea;
