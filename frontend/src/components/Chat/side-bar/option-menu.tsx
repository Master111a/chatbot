'use client';

import { cn } from '@/utils/cn';
import {
  forwardRef,
  ReactNode,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';
import { createPortal } from 'react-dom';

interface OptionMenuProps {
  parentElement: ReactNode;
  children: ReactNode;
  classNameContent?: string;
}

const OptionMenu = forwardRef(
  ({ parentElement, children, classNameContent }: OptionMenuProps, ref) => {
    const [isOpen, setIsOpen] = useState(false);
    const [position, setPosition] = useState({ top: 0, left: 0 });
    const triggerRef = useRef<HTMLDivElement>(null);
    const contentRef = useRef<HTMLDivElement>(null);

    useImperativeHandle(ref, () => ({
      onclose: () => setIsOpen(false),
    }));

    // const handleBlur = (e: React.FocusEvent) => {
    //   if (
    //     contentRef.current &&
    //     !contentRef.current.contains(e.relatedTarget as Node)
    //   ) {
    //     setIsOpen(false);
    //   }
    // };

    useEffect(() => {
      const handleClickOutside = (e: MouseEvent) => {
        if (
          isOpen &&
          contentRef.current &&
          !contentRef.current.contains(e.target as Node)
        ) {
          setIsOpen(false);
        }
      };

      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }, [isOpen]);

    useLayoutEffect(() => {
      if (isOpen && triggerRef.current) {
        const rect = triggerRef.current.getBoundingClientRect();
        setPosition({ top: rect.bottom + 4, left: rect.left });
      }
    }, [isOpen]);

    return (
      <div className="relative">
        <div
          ref={triggerRef}
          onClick={() => setIsOpen(true)}
          className="cursor-pointer"
          tabIndex={0}
          // onBlur={handleBlur}
        >
          {parentElement}
        </div>
        {isOpen &&
          typeof window !== 'undefined' &&
          createPortal(
            <div
              ref={contentRef}
              style={{
                position: 'fixed',
                top: position.top,
                left: position.left,
              }}
              className={cn(
                'bg-[var(--menu-bg)] shadow-sm z-[9999] p-2 rounded-16',
                classNameContent
              )}
            >
              {children}
            </div>,
            document.body
          )}
      </div>
    );
  }
);

OptionMenu.displayName = 'OptionMenu';

export default OptionMenu;
