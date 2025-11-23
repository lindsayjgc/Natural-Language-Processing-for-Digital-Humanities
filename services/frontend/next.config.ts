import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  output: 'standalone',
  experimental: {
    // Disable turbopack for more stable builds in Docker
    turbo: {
      resolveAlias: {
        // Add any alias configurations here if needed
      }
    }
  }
};

export default nextConfig;
