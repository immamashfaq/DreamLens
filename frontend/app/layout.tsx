import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ClerkProvider } from '@clerk/nextjs'; // 1. Import Clerk here

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "DreamLens",
  description: "AI Sleep Analysis Dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // 2. Wrap the <html> tag in the ClerkProvider
    <ClerkProvider> 
      <html lang="en">
        <body className={inter.className}>{children}</body>
      </html>
    </ClerkProvider>
  );
}