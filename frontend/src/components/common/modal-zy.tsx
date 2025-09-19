'use client';

import { X } from 'lucide-react';
import React, {
  forwardRef,
  useImperativeHandle,
  useState,
  useEffect,
} from 'react';
import ReactDOM from 'react-dom';
import { ButtonZy } from '../ui/button-zy';
import { cn } from '@/utils/cn';

interface PopupRef {
  openPopup: () => void;
  closePopup: () => void;
}

interface PopupComponentProps {
  children: React.ReactNode;
  title: string;
  classNameTitle?: string;
  classNameTextTitle?: string;
  classNamePopup?: string;
  classNameContent?: string;
}

const ModalZy = forwardRef<PopupRef, PopupComponentProps>(
  (
    {
      children,
      title,
      classNameTitle = '',
      classNameTextTitle = 'text-black',
      classNamePopup = '',
      classNameContent = '',
    },
    ref
  ) => {
    const [isOpen, setIsOpen] = useState(false);
    const [mounted, setMounted] = useState(false);

    useImperativeHandle(ref, () => ({
      openPopup: () => setIsOpen(true),
      closePopup: () => setIsOpen(false),
    }));

    // To ensure document is defined (for SSR safety)
    useEffect(() => {
      setMounted(true);
    }, []);

    if (!isOpen || !mounted) return null;

    const popupContent = (
      <div
        className="fixed h-[100vh] top-0 left-0 right-0 bottom-0 bg-black/80 bg-opacity-50 flex items-center justify-center z-[99998]"
        onClick={() => setIsOpen(false)}
      >
        <div
          className={cn(
            'relative min-w-[300px] max-w-[500px] bg-[var(--background)] w-full rounded-16 overflow-hidden',
            classNamePopup
          )}
          onClick={(e) => e.stopPropagation()}
        >
          <div
            className={cn(
              'flex justify-between items-center px-[21px] py-[18px]',
              classNameTitle
            )}
          >
            <div>
              <h1
                className={cn(
                  'text-20 font-700 text-black dark:text-white',
                  classNameTextTitle
                )}
              >
                {title}
              </h1>
            </div>
            <ButtonZy
              variant="text"
              className={` ${classNameTextTitle}`}
              onClick={() => setIsOpen(false)}
            >
              <X size={18} />
            </ButtonZy>
          </div>
          <div
            className={cn(
              'px-[21px] py-[18px] max-h-[80vh] overflow-y-auto',
              classNameContent
            )}
          >
            {children}
          </div>
        </div>
      </div>
    );

    return ReactDOM.createPortal(popupContent, document.body);
  }
);

ModalZy.displayName = 'ModalZy';

export default ModalZy;
