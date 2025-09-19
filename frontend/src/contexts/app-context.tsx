'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';

interface AppContextType {
  isSidebarOpen: boolean;
  setIsSidebarOpen: (isSidebarOpen: boolean) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

interface IAppProviderProps {
  children: React.ReactNode;
}

export const AppProvider = ({ children }: IAppProviderProps) => {
  const [isReady, setIsReady] = useState(false); // wait for localStorage load
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem('sidebarOpen');
    if (stored !== null) {
      setIsSidebarOpen(stored === 'true');
    }
    setIsReady(true);
  }, []);

  useEffect(() => {
    if (isReady) {
      localStorage.setItem('sidebarOpen', String(isSidebarOpen));
    }
  }, [isSidebarOpen, isReady]);

  if (!isReady) return null;

  return (
    <AppContext.Provider value={{ isSidebarOpen, setIsSidebarOpen }}>
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useApp must be used within a AppProvider');
  }
  return context;
};
