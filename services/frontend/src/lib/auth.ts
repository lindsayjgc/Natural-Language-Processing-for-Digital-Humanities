export interface User {
  id: string;
  email: string;
  created_at: string;
}

export interface LoginCredentials {
  email: string;
  password: string;
  remember_me?: boolean;
}

export interface RegisterData {
  email: string;
  password: string;
  remember_me?: boolean;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
}

// Token management
export const setAuthToken = (token: string) => {
  if (typeof window !== "undefined") {
    localStorage.setItem("auth_token", token);
  }
};

export const getAuthToken = (): string | null => {
  if (typeof window !== "undefined") {
    return localStorage.getItem("auth_token");
  }
  return null;
};

export const removeAuthToken = () => {
  if (typeof window !== "undefined") {
    localStorage.removeItem("auth_token");
  }
};

// API calls for authentication
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const authApi = {
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(credentials),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Login failed");
    }

    return response.json();
  },

  async register(userData: RegisterData): Promise<User> {
    const response = await fetch(`${API_BASE_URL}/auth/register`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(userData),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Registration failed");
    }

    return response.json();
  },

  async getCurrentUser(): Promise<User> {
    const token = getAuthToken();
    if (!token) {
      throw new Error("No authentication token");
    }

    // Create abort controller for timeout (fallback for older browsers)
    const controller = new AbortController();
    let timeoutId: number | null = setTimeout(() => controller.abort(), 10000); // 10 second timeout

    try {
      const response = await fetch(`${API_BASE_URL}/auth/me`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
        signal: controller.signal,
      });

      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }

      if (!response.ok) {
        if (response.status === 401) {
          removeAuthToken();
          throw new Error("Authentication expired");
        }
        if (response.status === 503) {
          // Database unavailable - don't remove token, but indicate service unavailable
          throw new Error("Database service unavailable. Please try again later.");
        }
        const errorData = await response.json().catch(() => ({ detail: "Failed to get user info" }));
        throw new Error(errorData.detail || "Failed to get user info");
      }

      return response.json();
    } catch (error) {
      // Clear timeout if it hasn't fired yet
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      
      // Handle network errors (backend not running, timeout, etc.)
      if (error instanceof TypeError) {
        throw new Error("Cannot connect to server. Please ensure the backend is running.");
      }
      if (error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError")) {
        throw new Error("Request timed out. The server may be unavailable.");
      }
      throw error;
    }
  },
};
