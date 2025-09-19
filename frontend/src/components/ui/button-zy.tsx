"use client";

import { cn } from "@/utils/cn";

interface IProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: "primary" | "outline" | "text";
    error?: boolean;
    disabled?: boolean;
}

const ButtonZy = ({
    variant = "primary",
    error,
    disabled = false,
    ...props
}: IProps) => {
    return (
        <button
            {...props}
            disabled={disabled}
            className={cn(
                "rounded-8 px-3 py-2 dark:text-white cursor-pointer transition-all duration-300 flex items-center gap-2",
                variant === "primary" &&
                    "bg-primary hover:bg-primary/80 text-white dark:text-black",
                variant === "outline" &&
                    "bg-secondary border border-gray-200 dark:border-gray-200/30 hover:bg-white/10",
                variant === "text" &&
                    "bg-transparent text-primary hover:bg-primary/10",
                error && "hover:bg-red-200 text-red-600 dark:hover:bg-red-500",
                props.className,
                disabled && "opacity-50 cursor-not-allowed"
            )}>
            {props.children}
        </button>
    );
};

export { ButtonZy };
