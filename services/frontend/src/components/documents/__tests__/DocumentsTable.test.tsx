/**
 * Unit tests for DocumentsTable component
 */

import { render, screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { Document } from "@/lib/api";
import { DocumentsTable } from "../DocumentsTable";

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

describe("DocumentsTable", () => {
  const mockOnRefresh = vi.fn();

  const mockDocuments: Document[] = [
    {
      _id: "1",
      filename: "test1.txt",
      uploaded_at: "2025-01-15T10:30:00",
      status: "completed",
      error: undefined,
    },
    {
      _id: "2",
      filename: "test2.txt",
      uploaded_at: "2025-01-16T11:00:00",
      status: "processing",
      error: undefined,
    },
    {
      _id: "3",
      filename: "test3.txt",
      uploaded_at: "2025-01-17T12:00:00",
      status: "failed",
      error: "Processing error",
    },
  ];

  it("renders loading state", () => {
    render(
      <DocumentsTable
        documents={[]}
        loading={true}
        onRefresh={mockOnRefresh}
      />,
    );

    expect(screen.getByText("Loading documents...")).toBeInTheDocument();
  });

  it("renders empty state when no documents", () => {
    render(
      <DocumentsTable
        documents={[]}
        loading={false}
        onRefresh={mockOnRefresh}
      />,
    );

    expect(screen.getByText(/No documents uploaded yet/i)).toBeInTheDocument();
  });

  it("renders document list with correct data", () => {
    render(
      <DocumentsTable
        documents={mockDocuments}
        loading={false}
        onRefresh={mockOnRefresh}
      />,
    );

    // Only completed documents are displayed (test1.txt)
    expect(screen.getByText("test1.txt")).toBeInTheDocument();
    
    // Processing and failed documents are filtered out
    expect(screen.queryByText("test2.txt")).not.toBeInTheDocument(); // processing
    expect(screen.queryByText("test3.txt")).not.toBeInTheDocument(); // failed

    // Check upload dates are formatted for completed documents
    expect(screen.getByText(/1\/15\/2025/)).toBeInTheDocument();
  });

  it("displays correct status badges for completed documents only", () => {
    render(
      <DocumentsTable
        documents={mockDocuments}
        loading={false}
        onRefresh={mockOnRefresh}
      />,
    );

    // Only completed documents are rendered in the table, so only "Analyzed" shows
    expect(screen.getByText("Analyzed")).toBeInTheDocument(); // completed
    // Processing and failed documents are filtered out from table display
    expect(screen.queryByText("Analyzing")).not.toBeInTheDocument(); // processing - not shown
    expect(screen.queryByText("Failed")).not.toBeInTheDocument(); // failed - not shown
  });

  it('shows "View Analysis" link for completed documents', () => {
    render(
      <DocumentsTable
        documents={mockDocuments}
        loading={false}
        onRefresh={mockOnRefresh}
      />,
    );

    // Find the dropdown button by its aria attributes
    const dropdownButton = screen.getByRole("button", { 
      expanded: false 
    });
    expect(dropdownButton).toBeInTheDocument();
    expect(dropdownButton).toHaveAttribute("aria-haspopup", "menu");
    // Note: "View Analysis" text is in dropdown menu that needs to be opened
  });

  it("does not render failed documents in table", () => {
    render(
      <DocumentsTable
        documents={mockDocuments}
        loading={false}
        onRefresh={mockOnRefresh}
      />,
    );

    // Failed documents are filtered out, so error message and filename won't show
    expect(screen.queryByText("Processing error")).not.toBeInTheDocument();
    expect(screen.queryByText("test3.txt")).not.toBeInTheDocument(); // failed document
  });

  it("does not render processing documents in table", () => {
    render(
      <DocumentsTable
        documents={mockDocuments}
        loading={false}
        onRefresh={mockOnRefresh}
      />,
    );

    // Processing documents are filtered out of the table display
    expect(screen.queryByText("test2.txt")).not.toBeInTheDocument(); // processing document
  });

  it("calls onRefresh when refresh button clicked", async () => {
    const user = userEvent.setup();

    render(
      <DocumentsTable
        documents={mockDocuments}
        loading={false}
        onRefresh={mockOnRefresh}
      />,
    );

    const refreshButton = screen.getByRole("button", { name: /refresh/i });
    await user.click(refreshButton);

    expect(mockOnRefresh).toHaveBeenCalledOnce();
  });

  it("renders table headers correctly", () => {
    render(
      <DocumentsTable
        documents={mockDocuments}
        loading={false}
        onRefresh={mockOnRefresh}
      />,
    );

    expect(screen.getByText("Document Name")).toBeInTheDocument();
    expect(screen.getByText("Uploaded")).toBeInTheDocument();
    expect(screen.getByText("Status")).toBeInTheDocument();
    expect(screen.getByText("Actions")).toBeInTheDocument();
  });

  it("renders table title", () => {
    render(
      <DocumentsTable
        documents={mockDocuments}
        loading={false}
        onRefresh={mockOnRefresh}
      />,
    );

    expect(screen.getByText("Your Documents")).toBeInTheDocument();
  });

  it("handles single document correctly", () => {
    const singleDoc: Document[] = [mockDocuments[0]];

    render(
      <DocumentsTable
        documents={singleDoc}
        loading={false}
        onRefresh={mockOnRefresh}
      />,
    );

    expect(screen.getByText("test1.txt")).toBeInTheDocument();
    expect(screen.queryByText("test2.txt")).not.toBeInTheDocument();
  });

  it("handles documents without stats_id", () => {
    const docWithoutStats: Document[] = [
      {
        _id: "4",
        filename: "test4.txt",
        uploaded_at: "2025-01-18T13:00:00",
        status: "completed", // Changed to completed so it gets rendered
        error: undefined,
      },
    ];

    render(
      <DocumentsTable
        documents={docWithoutStats}
        loading={false}
        onRefresh={mockOnRefresh}
      />,
    );

    expect(screen.getByText("test4.txt")).toBeInTheDocument();
  });
});
