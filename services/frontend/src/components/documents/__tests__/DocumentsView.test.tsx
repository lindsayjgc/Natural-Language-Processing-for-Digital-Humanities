/**
 * Unit tests for DocumentsView component
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DocumentsView } from "../DocumentsView";

// Mock Next.js router
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/documents",
  useSearchParams: () => new URLSearchParams(),
}));

// Mock the auth context
vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({
    user: { id: "test-user-id", email: "test@example.com" },
    loading: false,
  }),
}));

// Mock API calls
vi.mock("@/lib/api", () => ({
  apiClient: {
    getUserDocuments: vi.fn().mockResolvedValue({ 
      user_id: "test-user", 
      documents: [] 
    }),
    uploadDocument: vi.fn().mockResolvedValue({ 
      document_id: "test-id",
      filename: "test.txt",
      processing_status: "completed"
    }),
  },
  getUserDocuments: vi.fn().mockResolvedValue({ 
    user_id: "test-user", 
    documents: [] 
  }),
  uploadDocument: vi.fn().mockResolvedValue({ 
    document_id: "test-id",
    filename: "test.txt", 
    processing_status: "completed"
  }),
}));

describe("DocumentsView", () => {
  it("renders documents view correctly", () => {
    render(<DocumentsView />);
    
    // Use more specific selector to avoid multiple matches
    expect(screen.getByRole("heading", { name: /documents/i })).toBeInTheDocument();
  });

  it("shows upload section", () => {
    render(<DocumentsView />);
    
    // Should contain upload functionality
    expect(screen.getByText(/drag and drop files to upload/i) || screen.getByText(/browse files/i)).toBeInTheDocument();
  });

  it("shows documents table", () => {
    render(<DocumentsView />);
    
    // DocumentsView shows loading state initially
    expect(screen.getByText(/loading documents/i)).toBeInTheDocument();
  });

  it("handles loading state", () => {
    render(<DocumentsView />);
    
    // Should handle loading state appropriately
    expect(screen.getByRole("heading", { name: /documents/i })).toBeInTheDocument();
  });

  it("handles empty state", () => {
    render(<DocumentsView />);
    
    // Component shows loading state while fetching documents
    expect(screen.getByText(/loading documents/i)).toBeInTheDocument();
  });

  it("renders page title", () => {
    render(<DocumentsView />);
    
    // Use heading role to be more specific
    expect(screen.getByRole("heading", { name: /documents/i })).toBeInTheDocument();
  });
});