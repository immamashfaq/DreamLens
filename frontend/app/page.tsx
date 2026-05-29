'use client';

// Using the Clerk hooks and the Show component for conditional rendering
import { useClerk, Show } from '@clerk/nextjs';
import Link from 'next/link';

export default function Home() {
  // Accessing the Clerk modal methods for custom auth triggers
  const { openSignIn, openSignUp } = useClerk();

  return (
    <main className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-slate-950 text-slate-200">
      
      {/* Background visual effects - keeping the glowing orb theme for depth */}
      <div className="absolute top-1/4 left-1/4 h-96 w-96 rounded-full bg-blue-600/20 blur-[100px]"></div>
      <div className="absolute bottom-1/4 right-1/4 h-96 w-96 rounded-full bg-purple-600/20 blur-[100px]"></div>

      {/* Main Glassmorphism Card for the hero section */}
      <div className="relative z-10 flex flex-col items-center rounded-3xl border border-white/10 bg-white/5 p-12 text-center shadow-2xl backdrop-blur-xl md:p-16">
        
        <h1 className="mb-4 text-5xl font-extrabold tracking-tight text-white sm:text-7xl">
          Dream<span className="text-blue-400">Lens</span>
        </h1>
        
        <p className="mb-8 max-w-md text-lg text-slate-300">
          Your AI-powered EEG analysis platform. Decode your sleep stages and unlock personalized health insights.
        </p>
        
        <div className="flex w-full flex-col space-y-4 sm:flex-row sm:space-x-4 sm:space-y-0 sm:justify-center">
          
          {/* Logic to show these buttons only if the user is not currently logged in */}
          <Show when="signed-out">
            <button 
              onClick={() => openSignUp({ fallbackRedirectUrl: "/dashboard" })}
              className="w-full sm:w-auto rounded-full bg-blue-500 px-8 py-3 font-semibold text-white transition-all hover:scale-105 hover:bg-blue-600 shadow-[0_0_20px_rgba(59,130,246,0.4)]"
            >
              Get Started
            </button>

            <button 
              onClick={() => openSignIn({ fallbackRedirectUrl: "/dashboard" })}
              className="w-full sm:w-auto rounded-full border border-white/20 bg-transparent px-8 py-3 font-semibold text-white transition-all hover:bg-white/10"
            >
              Login
            </button>
          </Show>

          {/* Logic to show the dashboard link only if the user is already authenticated */}
          <Show when="signed-in">
            <Link 
              href="/dashboard"
              className="w-full sm:w-auto rounded-full bg-white px-8 py-3 font-semibold text-black transition-all hover:scale-105 hover:bg-gray-200 shadow-[0_0_20px_rgba(255,255,255,0.3)]"
            >
              Go to Dashboard
            </Link>
          </Show>
          
        </div>
      </div>
    </main>
  );
}