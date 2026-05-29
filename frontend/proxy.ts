import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

// This protects the dashboard, but leaves the homepage ("/") public
const isProtectedRoute = createRouteMatcher(["/dashboard(.*)"]);

export default clerkMiddleware(async (auth, req) => {
  if (isProtectedRoute(req)) await auth.protect();
});

export const config = {
  matcher: ["/((?!.*\\..*|_next).*)", "/", "/(api|trpc)(.*)"],
};