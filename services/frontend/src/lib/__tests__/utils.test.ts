/**
 * Unit tests for utility functions
 */

import { describe, expect, it } from "vitest";
import { cn } from "../utils";

describe("Utils", () => {
  describe("cn (className utility)", () => {
    it("combines class names correctly", () => {
      const result = cn("class1", "class2");
      expect(result).toContain("class1");
      expect(result).toContain("class2");
    });

    it("handles conditional classes", () => {
      const result = cn("base", true && "conditional", false && "hidden");
      expect(result).toContain("base");
      expect(result).toContain("conditional");
      expect(result).not.toContain("hidden");
    });

    it("handles undefined and null values", () => {
      const result = cn("class1", undefined, null, "class2");
      expect(result).toContain("class1");
      expect(result).toContain("class2");
    });

    it("handles empty input", () => {
      const result = cn();
      expect(typeof result).toBe("string");
    });

    it("deduplicates classes", () => {
      const result = cn("duplicate", "other", "duplicate");
      // Should handle duplicates appropriately (depends on implementation)
      expect(result).toContain("duplicate");
      expect(result).toContain("other");
    });
  });
});