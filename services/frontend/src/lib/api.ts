// API client for NLP Document Library backend
import { getAuthToken } from "./auth";
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Document {
  _id: string;
  filename: string;
  uploaded_at: string;
  status: "processing" | "completed" | "failed";
  error?: string;
  stats?: DocumentStats;
}

export interface DocumentStats {
  vocab_size: number;
  token_count: number;
  type_token_ratio: number;
  doc_sentiment: {
    emotions: Record<string, number>;
    dominant_emotion: string;
    confidence: number;
  };
  sentiment_method: string;
}

export interface UserDocuments {
  user_id: string;
  documents: Document[];
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  processing_status: string;
  stats?: DocumentStats;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const token = getAuthToken();

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(options.headers as Record<string, string>),
    };

    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error: ${response.status} - ${errorText}`);
    }

    return response.json();
  }

  // Get all documents for a user
  async getUserDocuments(userId: string): Promise<UserDocuments> {
    return this.request<UserDocuments>(`/documents/${userId}`);
  }

  // Get a specific document with stats
  async getDocument(userId: string, documentId: string): Promise<Document> {
    return this.request<Document>(`/documents/${userId}/${documentId}`);
  }

  // Upload a document
  async uploadDocument(
    userId: string,
    file: File,
    _onProgress?: (progress: number) => void,
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append("user_id", userId);
    formData.append("file", file);

    const token = getAuthToken();
    const headers: Record<string, string> = {};
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }

    try {
      const response = await fetch(`${this.baseUrl}/documents/upload`, {
        method: "POST",
        headers,
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload Error: ${response.status} - ${errorText}`);
      }

      return response.json();
    } catch (error) {
      if (error instanceof TypeError && error.message.includes("fetch")) {
        throw new Error(
          "Failed to connect to server. Please check if the API is running.",
        );
      }
      throw error;
    }
  }

  // Health check
  async healthCheck(): Promise<{ message: string; version: string }> {
    return this.request<{ message: string; version: string }>("/");
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Export individual functions for convenience
export const { getUserDocuments, getDocument, uploadDocument, healthCheck } =
  apiClient;
