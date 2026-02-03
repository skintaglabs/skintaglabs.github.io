import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  basePath: '/SkinTag/app',
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
