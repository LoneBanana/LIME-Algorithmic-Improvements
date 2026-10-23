import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ArcTanh-LIME",
  description:
    "LIME rebuilt from scratch, then given a bounded tanh⁻¹ surrogate whose asymptotes come from walking the black box to its nearest local extrema — with interactive demos and honest benchmarks.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

// Applies a stored light/dark choice before first paint (no flash).
const themeScript =
  "try{var t=localStorage.getItem('atlime-theme');if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t)}catch(e){}";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
