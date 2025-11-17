"use client";

import type React from "react";
import { createContext, useContext, useEffect, useState } from "react";
import {
  authApi,
  getAuthToken,
  type LoginCredentials,
  type RegisterData,
  removeAuthToken,
  setAuthToken,
  type User,
} from "@/lib/auth";

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (userData: RegisterData) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const isAuthenticated = !!user;

  const login = async (credentials: LoginCredentials) => {
    try {
      const response = await authApi.login(credentials);
      setAuthToken(response.access_token);
      await refreshUser();
    } catch (error) {
      throw error;
    }
  };

  const register = async (userData: RegisterData) => {
    try {
      await authApi.register(userData);
      // After successful registration, log the user in
      await login({
        email: userData.email,
        password: userData.password,
        remember_me: userData.remember_me,
      });
    } catch (error) {
      throw error;
    }
  };

  const logout = () => {
    removeAuthToken();
    setUser(null);
  };

  const refreshUser = async () => {
    try {
      const token = getAuthToken();
      if (!token) {
        setUser(null);
        return;
      }

      const userData = await authApi.getCurrentUser();
      setUser(userData);
    } catch (error) {
      // Only log non-network errors to avoid console spam
      const errorMessage = error instanceof Error ? error.message : String(error);
      if (!errorMessage.includes("connect to server") && 
          !errorMessage.includes("timed out") && 
          !errorMessage.includes("Database service unavailable")) {
        console.error("Failed to refresh user:", error);
      }
      // Clear user on any error - they'll need to log in again
      setUser(null);
      // Don't remove token on network/database errors - might be temporary
      // Only remove token on authentication errors
      if (errorMessage.includes("Authentication expired") || errorMessage.includes("401")) {
        removeAuthToken();
      }
    }
  };

  // Check for existing token on mount
  useEffect(() => {
    let isMounted = true;
    
    const initializeAuth = async () => {
      try {
        const token = getAuthToken();
        if (!token) {
          // No token, set loading to false immediately
          if (isMounted) {
            setIsLoading(false);
          }
          return;
        }

        // Token exists, try to refresh user
        // refreshUser handles its own errors internally, so we don't need nested try-catch
        await refreshUser();
      } catch (error) {
        // Catch any unexpected errors
        console.error("Unexpected error during auth initialization:", error);
      } finally {
        // Always set loading to false, even if something went wrong
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    // Add a safety timeout to ensure loading always stops
    // Use a ref to track if initialization has completed
    let initializationComplete = false;
    const timeoutId = setTimeout(() => {
      if (isMounted && !initializationComplete) {
        console.warn("Auth initialization timeout - forcing loading to complete");
        setIsLoading(false);
      }
    }, 15000); // 15 second max timeout

    initializeAuth().finally(() => {
      initializationComplete = true;
      clearTimeout(timeoutId);
    });

    return () => {
      isMounted = false;
      clearTimeout(timeoutId);
    };
  }, []);

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated,
    login,
    register,
    logout,
    refreshUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
