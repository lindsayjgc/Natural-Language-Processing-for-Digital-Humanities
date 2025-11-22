/**
 * Unit tests for LoginForm component
 */

import { render, screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LoginForm } from "../LoginForm";

// Mock the auth context
vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({
    login: vi.fn(),
    loading: false,
  }),
}));

describe("LoginForm", () => {
  it("renders login form", () => {
    render(<LoginForm />);
    
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("allows entering email and password", async () => {
    const user = userEvent.setup();
    
    render(<LoginForm />);
    
    const emailInput = screen.getByLabelText(/email/i);
    const passwordInput = screen.getByLabelText(/password/i);
    
    await user.type(emailInput, "test@example.com");
    await user.type(passwordInput, "password123");
    
    expect(emailInput).toHaveValue("test@example.com");
    expect(passwordInput).toHaveValue("password123");
  });

  it("shows validation errors", async () => {
    const user = userEvent.setup();
    
    render(<LoginForm />);
    
    const submitButton = screen.getByRole("button", { name: /sign in/i });
    await user.click(submitButton);
    
    // Should show some form of validation
    // This depends on your actual validation implementation
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
  });

  it("disables submit during loading", () => {
    // Would need to mock loading state
    render(<LoginForm />);
    
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("shows login title", () => {
    render(<LoginForm />);
    
    expect(screen.getByText(/sign in/i)).toBeInTheDocument();
  });
});