import { useEffect, useState } from 'react';

export type Breakpoint = 'xs' | 'sm' | 'md' | 'lg' | 'xl';

const getBreakpoint = (width: number): Breakpoint => {
  if (width < 640) return 'xs'; // Tailwind: <sm
  if (width < 768) return 'sm'; // Tailwind: sm
  if (width < 1024) return 'md'; // Tailwind: md
  if (width < 1280) return 'lg'; // Tailwind: lg
  return 'xl'; // Tailwind: xl and up
};

export const useResponsive = () => {
  const [breakpoint, setBreakpoint] = useState<Breakpoint | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handleResize = () => {
      setBreakpoint(getBreakpoint(window.innerWidth));
    };

    handleResize(); // Set initial breakpoint

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const is = (bp: Breakpoint) => breakpoint === bp;

  return {
    breakpoint,
    isXs: is('xs'),
    isSm: is('sm'),
    isMd: is('md'),
    isLg: is('lg'),
    isXl: is('xl'),
    isReady: breakpoint !== null,
  };
};
