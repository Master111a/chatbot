import { ButtonZy } from "@/components/ui/button-zy";
import appPath from "@/constants/app-path";
import { useApp } from "@/contexts/app-context";
import { useChat } from "@/contexts/chat-context";
import { useTheme } from "@/contexts/theme-context";
import { Menu, Moon, Sun } from "lucide-react";
import Image from "next/image";
import Link from "next/link";

export const Topbar = () => {
    const { currentChat } = useChat();
    const { theme, toggleTheme } = useTheme();
    const { isSidebarOpen, setIsSidebarOpen } = useApp();

    return (
        <div className="flex items-center justify-between bg-white dark:bg-transparent theme-transition duration-500 h-14 px-[15px] text-14">
            <div className="flex items-center gap-2">
                {!isSidebarOpen && (
                    <ButtonZy
                        variant="text"
                        onClick={() => setIsSidebarOpen(true)}
                        className="text-primary">
                        <Menu size={22} />
                    </ButtonZy>
                )}
                <div className="px-3">
                    <Link href={appPath.home}>
                        <ButtonZy
                            variant="text"
                            className="text-primary w-full h-11">
                            <div className="flex items-center gap-2">
                                {theme !== "dark" ? (
                                    <Image
                                        src="/logo-nws-light.png"
                                        alt="Newwave Solutions - a top-notch Software Development Company providing optimized Digital Transformation Services."
                                        width={274}
                                        height={72}
                                        className="w-[137px] h-9"
                                    />
                                ) : (
                                    <Image
                                        src="/logo-nws.png"
                                        alt="Newwave Solutions - a top-notch Software Development Company providing optimized Digital Transformation Services."
                                        width={274}
                                        height={72}
                                        className="w-[137px] h-9"
                                    />
                                )}
                                <div className="relative flex size-2 translate-y-[-8px]">
                                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary dark:bg-primary opacity-75"></span>
                                    <span className="relative inline-flex size-2 rounded-full bg-primary dark:bg-primary"></span>
                                </div>
                            </div>
                        </ButtonZy>
                    </Link>
                </div>
            </div>
            <div className="flex items-center gap-2 px-3 py-2 text-primary">
                <span className="font-600 max-w-[200px] truncate">
                    {currentChat?.title}
                </span>
            </div>
            <div className="flex items-center gap-2">
                <ButtonZy
                    variant="text"
                    onClick={toggleTheme}
                    className="text-gray-600 dark:text-sky-300"
                    title={
                        theme === "dark"
                            ? "Switch to Light Mode"
                            : "Switch to Dark Mode"
                    }>
                    {theme === "dark" ? <Moon size={20} /> : <Sun size={20} />}
                </ButtonZy>
            </div>
        </div>
    );
};
