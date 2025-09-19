import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  /* config options here */
  reactStrictMode: false,
  
  // Environment variables configuration
  env: {
    ENV: process.env.ENV,
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },
  
  // Output configuration for Docker
  output: 'standalone',
};

export default nextConfig;
