import { useEffect, useRef } from "react";
import { Control, Controller, useWatch } from "react-hook-form";

interface ChatInputProps {
    name: string;
    control: Control<any>;
    onEnterPress: (e: React.FormEvent) => void;
}

export default function ChatInput({
    name,
    control,
    onEnterPress,
}: ChatInputProps) {
    const textareaRef = useRef<HTMLTextAreaElement | null>(null);
    const value = useWatch({ control, name });

    useEffect(() => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = "auto";
            const newHeight = Math.min(textarea.scrollHeight, 185);
            textarea.style.height = `${newHeight}px`;
            textarea.style.overflowY = newHeight === 185 ? "auto" : "hidden";
        }
    }, [value]);

    useEffect(() => {
        textareaRef.current?.focus();
    }, []);

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.metaKey && !e.shiftKey) {
            e.preventDefault();
            onEnterPress(e);
        }
    };

    return (
        <Controller
            name={name}
            control={control}
            render={({ field }) => (
                <textarea
                    {...field}
                    ref={(el) => {
                        textareaRef.current = el;
                        field.ref(el);
                    }}
                    onKeyDown={handleKeyDown}
                    rows={1}
                    placeholder="Ask Newwave AI..."
                    className="w-full p-4 pr-14 bg-transparent text-[var(--text-primary)] rounded-lg focus:outline-none resize-none"
                />
            )}
        />
    );
}
