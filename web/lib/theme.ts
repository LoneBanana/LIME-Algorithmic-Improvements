"use client";

import { useEffect, useState } from "react";

export const THEME_EVENT = "atlime-themechange";

/** Resolved values of CSS custom properties, refreshed when the theme changes (for canvas drawing). */
export function useCssColors<T extends string>(names: readonly T[]): Record<T, string> | null {
  const [colors, setColors] = useState<Record<T, string> | null>(null);
  const key = names.join(",");
  useEffect(() => {
    const read = () => {
      const cs = getComputedStyle(document.documentElement);
      const out = {} as Record<T, string>;
      for (const n of key.split(",") as T[]) out[n] = cs.getPropertyValue(`--${n}`).trim();
      setColors(out);
    };
    read();
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", read);
    window.addEventListener(THEME_EVENT, read);
    return () => {
      mq.removeEventListener("change", read);
      window.removeEventListener(THEME_EVENT, read);
    };
  }, [key]);
  return colors;
}

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return reduced;
}
