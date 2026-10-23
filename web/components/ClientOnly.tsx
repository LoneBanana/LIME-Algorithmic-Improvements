"use client";

import { useEffect, useState } from "react";

/**
 * Renders its children only in the browser. The interactive figures compute
 * their geometry from floating-point maths whose last digits differ between the
 * build machine's and the browser's JS engines (and from the measured width),
 * so server-rendering them would only produce hydration mismatches.
 */
export default function ClientOnly({ children, minHeight }: { children: React.ReactNode; minHeight: number }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) {
    return <div className="card" style={{ minHeight, margin: "28px 0" }} aria-busy="true" aria-label="Loading figure" />;
  }
  return <>{children}</>;
}
