"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import {
  User,
  LoginCredentials,
  RegisterData,
  authApi,
  getAuthToken,
  setAuthToken,
  removeAuthToken,
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
      await login({ email: userData.email, password: userData.password, remember_me: userData.remember_me });
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
      console.error("Failed to refresh user:", error);
      setUser(null);
    }
  };

  // Check for existing token on mount
  useEffect(() => {
    const initializeAuth = async () => {
      const token = getAuthToken();
      if (token) {
        try {
          await refreshUser();
        } catch (error) {
          console.error("Failed to initialize auth:", error);
          removeAuthToken();
        }
      }
      setIsLoading(false);
    };

    initializeAuth();
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
