import "@testing-library/jest-dom";

// Mock environment variables
process.env.NEXT_PUBLIC_API_URL = "http://localhost:8000";

// Mock ResizeObserver which is not available in test environment
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};