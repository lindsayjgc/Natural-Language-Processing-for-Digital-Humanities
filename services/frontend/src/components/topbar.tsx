"use client";

import Link from "next/link";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { UserMenu } from "@/components/auth/UserMenu";
import { AuthDialog } from "@/components/auth/AuthDialog";
import { useState } from "react";

export function TopBar() {
  const { isAuthenticated, isLoading } = useAuth();
  const [authDialogOpen, setAuthDialogOpen] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');

  const handleAuthClick = (mode: 'login' | 'register') => {
    setAuthMode(mode);
    setAuthDialogOpen(true);
  };

  return (
    <>
      <div className="w-full sticky top-0 z-50 bg-white/95 backdrop-blur border-b border-black/10">
        <div className="mx-auto max-w-6xl px-4 h-14 flex items-center gap-3">
          <Link href="/" className="flex items-center gap-2">
            <span className="text-gray-900 font-semibold">LitLens</span>
          </Link>
          <Separator
            decorative
            orientation="vertical"
            className="h-6 bg-black/10"
          />
          {isAuthenticated && (
            <nav className="hidden md:flex items-center gap-1 text-sm">
              <Link
                href="/documents"
                className="text-gray-600 hover:text-gray-900 px-3 py-1 rounded-md"
              >
                Documents
              </Link>
              <Link
                href="/stats"
                className="text-gray-600 hover:text-gray-900 px-3 py-1 rounded-md"
              >
                Stats
              </Link>
            </nav>
          )}
          <div className="ml-auto flex items-center gap-3">
            {isLoading ? (
              <div className="h-8 w-8 animate-pulse bg-gray-200 rounded-full" />
            ) : isAuthenticated ? (
              <UserMenu />
            ) : (
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleAuthClick('login')}
                >
                  Sign In
                </Button>
                <Button
                  size="sm"
                  onClick={() => handleAuthClick('register')}
                >
                  Sign Up
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
      <AuthDialog
        open={authDialogOpen}
        onOpenChange={setAuthDialogOpen}
        defaultMode={authMode}
      />
    </>
  );
}
