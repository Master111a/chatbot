"use client";

import OptionMenu from "@/components/Chat/side-bar/option-menu";
import ModalZy from "@/components/common/modal-zy";
import { ButtonZy } from "@/components/ui/button-zy";
import appPath from "@/constants/app-path";
import { Chat, useChat } from "@/contexts/chat-context";
import { cn } from "@/utils/cn";
import { Ellipsis, Pen, Trash2 } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import React, { useRef, useState } from "react";

interface PopupRef {
    openPopup: () => void;
    closePopup: () => void;
}

const ChatITem = ({ chat }: { chat: Chat }) => {
    const { slug } = useParams();
    const { selectChat, updateChatTitle, deleteChat } = useChat();
    const [editingChatId, setEditingChatId] = useState<string | null>(null);
    const [editingTitle, setEditingTitle] = useState("");

    const modalRef = useRef<PopupRef>(null);

    const handleEditClick = (chatId: string, currentTitle: string) => {
        setEditingChatId(chatId);
        setEditingTitle(currentTitle);
    };

    const handleSaveTitle = (chatId: string) => {
        if (editingTitle.trim() && editingTitle.trim() !== chat.title) {
            updateChatTitle(chatId, editingTitle.trim());
        }
        setEditingChatId(null);
    };

    const handleKeyPress = (e: React.KeyboardEvent, chatId: string) => {
        if (e.key === "Enter") {
            handleSaveTitle(chatId);
        } else if (e.key === "Escape") {
            setEditingChatId(null);
        }
    };
    return (
        <>
            <Link
                href={appPath.chatDetail(chat.session_id)}
                className={cn(
                    "flex items-center rounded-md justify-between p-2 hover:bg-black/10 dark:hover:bg-black/10 cursor-pointer h-11 transition-all duration-100 dark:text-primary",
                    slug === chat.session_id ? "bg-transparent" : ""
                )}
                onClick={() => {
                    selectChat(chat.session_id);
                }}
                onDoubleClick={() =>
                    handleEditClick(chat.session_id, chat.title)
                }>
                {editingChatId === chat.session_id ? (
                    <input
                        type="text"
                        value={editingTitle}
                        onChange={(e) => setEditingTitle(e.target.value)}
                        onBlur={() => {
                            handleSaveTitle(chat.session_id);
                        }}
                        onKeyDown={(e) => handleKeyPress(e, chat.session_id)}
                        className="flex-1 bg-transparent border-none outline-none rounded text-[var(--text-primary)]"
                        autoFocus
                    />
                ) : (
                    <div className="flex items-center gap-2 flex-1">
                        <span className="max-w-[180px] truncate text-14 text-gray-600 dark:text-primary">
                            {chat.title}
                        </span>
                    </div>
                )}
                {!editingChatId && (
                    <div className="flex items-center gap-1">
                        <div
                            className="text-gray-600 p-0"
                            onClick={(e) => {
                                /**
                                 * @description: Prevent the default behavior of the link
                                 * @description: Prevent the propagation of the event
                                 */
                                e.preventDefault();
                                e.stopPropagation();
                            }}
                            onDoubleClick={(e) => {
                                e.stopPropagation();
                            }}>
                            <OptionMenu
                                parentElement={
                                    <Ellipsis
                                        size={20}
                                        className="text-gray-600 dark:text-primary/80"
                                    />
                                }
                                classNameContent="top-6 right-0 w-32 border-[1px] border-gray-100 dark:border-[var(--menu-bg)]">
                                <ButtonZy
                                    variant="text"
                                    className="w-full h-full"
                                    onClick={() => {
                                        handleEditClick(
                                            chat.session_id,
                                            chat.title
                                        );
                                    }}>
                                    <Pen size={16} />
                                    <span>Edit</span>
                                </ButtonZy>
                                <ButtonZy
                                    variant="text"
                                    error={true}
                                    className="w-full h-full"
                                    onClick={() => {
                                        modalRef.current?.openPopup();
                                        // deleteChat(chat.session_id);
                                    }}>
                                    <Trash2 size={16} />
                                    <span>Delete</span>
                                </ButtonZy>
                            </OptionMenu>
                        </div>
                    </div>
                )}
            </Link>
            <ModalZy
                title="Delete Chat?"
                ref={modalRef}
                classNameContent="pt-0"
                classNamePopup="max-w-[420px]">
                <p className="text-14 dark:text-white/90">
                    Are you sure you want to delete{" "}
                    <span className="font-700">{chat.title}</span>?
                </p>
                <div className="flex justify-end items-center mt-4 gap-4">
                    <ButtonZy
                        variant="outline"
                        className="!text-14 rounded-16 font-600"
                        onClick={() => {
                            modalRef.current?.closePopup();
                        }}>
                        Cancel
                    </ButtonZy>
                    <ButtonZy
                        variant="primary"
                        className="!text-14 text-white dark:text-white bg-red-500 rounded-16 font-600 hover:bg-red-500/80"
                        onClick={() => deleteChat(chat.session_id)}>
                        Delete
                    </ButtonZy>
                </div>
            </ModalZy>
        </>
    );
};

export default ChatITem;
