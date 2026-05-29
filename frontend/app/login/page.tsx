'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

export default function LoginPage() {
  // Using a boolean to swap between the login and signup forms
  const [isLogin, setIsLogin] = useState(true);

  // Check the URL when the component mounts
  // If the user clicked "Get Started" on the landing page, we show registration first
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('mode') === 'register') {
      setIsLogin(false); 
    }
  }, []);

  return (
    <main className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-slate-950 text-slate-200">
      
      {/* Decorative background orbs to match the DreamLens aesthetic */}
      <div className="absolute top-1/4 right-1/4 h-96 w-96 rounded-full bg-blue-600/10 blur-[100px]"></div>
      <div className="absolute bottom-1/4 left-1/4 h-96 w-96 rounded-full bg-purple-600/10 blur-[100px]"></div>

      {/* Quick link to head back to the landing page */}
      <div className="absolute top-8 left-8 z-20">
        <Link 
          href="/" 
          className="flex items-center gap-2 text-sm font-medium text-slate-400 transition-colors hover:text-white"
        >
          &larr; Back to Home
        </Link>
      </div>

      {/* Main glassmorphism card for the auth UI */}
      <div className="relative z-10 w-full max-w-md flex-col items-center rounded-3xl border border-white/10 bg-white/5 p-8 shadow-2xl backdrop-blur-xl sm:p-10">
        
        {/* Title and description update based on isLogin state */}
        <div className="mb-8 text-center">
          <h2 className="text-3xl font-bold text-white">
            {isLogin ? 'Welcome Back' : 'Create Account'}
          </h2>
          <p className="mt-2 text-sm text-slate-400">
            {isLogin 
              ? 'Enter your credentials to access your dashboard.' 
              : 'Join DreamLens to decode your sleep data today.'}
          </p>
        </div>

        {/* Auth form - using preventDefault for now since real auth logic comes later */}
        <form className="flex flex-col gap-4" onSubmit={(e) => e.preventDefault()}>
          
          {/* Conditional rendering: only show name input if we are in signup mode */}
          {!isLogin && (
            <div>
              <label className="mb-1 block text-sm font-medium text-slate-300">Full Name</label>
              <input 
                type="text" 
                placeholder="John Doe" 
                className="w-full rounded-lg border border-white/20 bg-white/5 px-4 py-3 text-white placeholder-slate-500 transition-colors focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500" 
              />
            </div>
          )}
          
          <div>
            <label className="mb-1 block text-sm font-medium text-slate-300">Email Address</label>
            <input 
              type="email" 
              placeholder="you@example.com" 
              className="w-full rounded-lg border border-white/20 bg-white/5 px-4 py-3 text-white placeholder-slate-500 transition-colors focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500" 
            />
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-slate-300">Password</label>
            <input 
              type="password" 
              placeholder="••••••••" 
              className="w-full rounded-lg border border-white/20 bg-white/5 px-4 py-3 text-white placeholder-slate-500 transition-colors focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500" 
            />
          </div>

          {/* Simple Link navigation to dashboard for the prototype demonstration */}
          <Link href="/dashboard" className="mt-4 w-full">
            <button className="w-full rounded-lg bg-blue-500 py-3 font-semibold text-white shadow-[0_0_15px_rgba(59,130,246,0.3)] transition-all hover:bg-blue-600">
              {isLogin ? 'Sign In' : 'Sign Up'}
            </button>
          </Link>
        </form>

        {/* Visual divider for social auth options */}
        <div className="my-6 flex items-center justify-center gap-3">
          <div className="h-px w-full bg-white/10"></div>
          <span className="text-xs uppercase tracking-wider text-slate-400">Or continue with</span>
          <div className="h-px w-full bg-white/10"></div>
        </div>

        {/* Placeholder buttons for future social integration (Oauth) */}
        <div className="flex gap-4">
          <button className="flex w-full items-center justify-center rounded-lg border border-white/20 bg-transparent py-2.5 text-sm font-medium text-white transition-colors hover:bg-white/10">
            Google
          </button>
          <button className="flex w-full items-center justify-center rounded-lg border border-white/20 bg-transparent py-2.5 text-sm font-medium text-white transition-colors hover:bg-white/10">
            GitHub
          </button>
        </div>

        {/* Toggling the view between Login and Signup */}
        <p className="mt-8 text-center text-sm text-slate-400">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button 
            onClick={() => setIsLogin(!isLogin)} 
            className="font-semibold text-blue-400 transition-colors hover:text-blue-300"
          >
            {isLogin ? 'Sign up' : 'Log in'}
          </button>
        </p>

      </div>
    </main>
  );
}