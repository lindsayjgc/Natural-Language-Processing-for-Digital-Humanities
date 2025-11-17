/**
 * Unit tests for API client
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiClient } from "../api";

// Mock fetch
global.fetch = vi.fn() as unknown as typeof fetch;

describe("API Client", () => {
  beforeEach(() => {
    // Reset mocks before each test
    vi.clearAllMocks();
  });

  describe("getUserDocuments", () => {
    it("should fetch documents for a user", async () => {
      const mockResponse = {
        user_id: "test_user",
        documents: [
          {
            _id: "123",
            filename: "test.txt",
            uploaded_at: "2025-01-15T10:30:00",
            status: "completed",
          },
        ],
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await apiClient.getUserDocuments("test_user");

      expect(fetch).toHaveBeenCalledWith(
        "http://localhost:8000/documents/test_user",
        expect.objectContaining({
          headers: expect.any(Object),
        }),
      );
      expect(result).toEqual(mockResponse);
    });

    it("should throw error on API failure", async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ detail: "Internal server error" }),
      });

      await expect(apiClient.getUserDocuments("test_user")).rejects.toThrow();
    });

    it("should throw error on network failure", async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error("Network error"));

      await expect(apiClient.getUserDocuments("test_user")).rejects.toThrow(
        "Network error",
      );
    });
  });

  describe("uploadDocument", () => {
    it("should upload a document with FormData", async () => {
      const mockFile = new File(["test content"], "test.txt", {
        type: "text/plain",
      });
      const mockResponse = {
        library_item_id: "123",
        filename: "test.txt",
        processing_status: "completed",
        stats: {
          vocab_size: 10,
          word_count: 20,
        },
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await apiClient.uploadDocument("test_user", mockFile);

      expect(fetch).toHaveBeenCalledWith(
        "http://localhost:8000/documents/upload",
        expect.objectContaining({
          method: "POST",
          body: expect.any(FormData),
        }),
      );

      expect(result).toEqual(mockResponse);
    });

    it("should handle upload errors", async () => {
      const mockFile = new File(["test"], "test.txt", { type: "text/plain" });

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: "Invalid file" }),
      });

      await expect(
        apiClient.uploadDocument("test_user", mockFile),
      ).rejects.toThrow();
    });
  });

  describe("getDocument", () => {
    it("should fetch a specific document with stats", async () => {
      const mockDocument = {
        _id: "123",
        user_id: "test_user",
        filename: "test.txt",
        uploaded_at: "2025-01-15T10:30:00",
        status: "completed",
        stats: {
          vocab_size: 100,
          word_count: 500,
          type_token_ratio: 0.2,
          doc_sentiment: {
            neutral: 0.7,
            joy: 0.2,
            sadness: 0.1,
          },
        },
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDocument,
      });

      const result = await apiClient.getDocument("test_user", "123");

      expect(fetch).toHaveBeenCalledWith(
        "http://localhost:8000/documents/test_user/123",
        expect.objectContaining({
          headers: expect.any(Object),
        }),
      );
      expect(result).toEqual(mockDocument);
      expect(result.stats).toBeDefined();
      expect(result.stats?.vocab_size).toBe(100);
    });

    it("should throw error for non-existent document", async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: "Document not found" }),
      });

      await expect(
        apiClient.getDocument("test_user", "nonexistent"),
      ).rejects.toThrow();
    });
  });

  describe("Error handling", () => {
    it("should handle 500 server errors", async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ detail: "Internal server error" }),
      });

      await expect(apiClient.getUserDocuments("test_user")).rejects.toThrow();
    });

    it("should handle malformed JSON responses", async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => {
          throw new Error("Invalid JSON");
        },
      });

      await expect(apiClient.getUserDocuments("test_user")).rejects.toThrow();
    });

    it("should handle timeout scenarios", async () => {
      vi.mocked(fetch).mockImplementationOnce(
        () =>
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error("Request timeout")), 100),
          ),
      );

      await expect(apiClient.getUserDocuments("test_user")).rejects.toThrow(
        "Request timeout",
      );
    });
  });

  describe("API URL configuration", () => {
    it("should use environment variable for API URL", () => {
      expect(process.env.NEXT_PUBLIC_API_URL).toBe("http://localhost:8000");
    });
  });
});
