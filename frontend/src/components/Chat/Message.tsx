"use client";

import { MessageType } from "@/types";
import React from "react";
import ReactMarkdown from "react-markdown";

interface MessageProps {
    content: MessageType;
}

export const Message: React.FC<MessageProps> = ({ content }) => {
    return (
        <div className="flex flex-col gap-10">
            {/* User message (right side) */}
            {content.message && (
                <div className="flex items-start justify-end gap-4 relative">
                    <div className="bg-[var(--background-message-user)] text-black dark:text-white px-5 py-[10px] rounded-2xl max-w-[538px] inline-block break-words whitespace-pre-wrap">
                        {content.message}
                    </div>
                </div>
            )}

            {/* AI response (left side) */}
            {content.response && (
                <div className="flex grow-1 items-start gap-4 relative">
                    <div className="prose dark:prose-invert max-w-[768px] w-full">
                        <ReactMarkdown
                            components={{
                                a: ({ href, children }) => (
                                    <a
                                        href={href}
                                        target="_blank"
                                        rel="noopener noreferrer">
                                        {children}
                                    </a>
                                ),
                            }}>
                            {content.response}
                        </ReactMarkdown>
                    </div>
                </div>
            )}
        </div>
    );
};
