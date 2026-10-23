import type { NextConfig } from "next";

// Fully static site: `npm run build` writes ./out (no server needed).
const nextConfig: NextConfig = {
  output: "export",
  reactStrictMode: true,
};

export default nextConfig;
