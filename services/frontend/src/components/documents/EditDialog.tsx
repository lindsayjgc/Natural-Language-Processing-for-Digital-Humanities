import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { Document } from "@/lib/api";

type EditDocumentDialogProps = {
  document: Document | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (documentId: string, newFilename: string) => Promise<void>;
};

export function EditDocumentDialog({
  document,
  open,
  onOpenChange,
  onSave,
}: EditDocumentDialogProps) {
  const [filename, setFilename] = useState("");
  const [loading, setLoading] = useState(false);

  useState(() => {
    if (document) {
      setFilename(document.filename);
    }
  });

  const handleSave = async () => {
    if (!document || !filename.trim()) return;

    setLoading(true);
    try {
      await onSave(document._id, filename.trim());
      onOpenChange(false);
    } catch (error) {
      console.error("Failed to update document:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenChange = (newOpen: boolean) => {
    if (!loading) {
      onOpenChange(newOpen);
      if (!newOpen) {
        setFilename(document?.filename || "");
      }
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Edit Document Name</DialogTitle>
          <DialogDescription>
            Update the name of your document. Click save when you're done.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="filename">Document Name</Label>
            <Input
              id="filename"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              placeholder="Enter document name"
              disabled={loading}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !loading) {
                  handleSave();
                }
              }}
            />
          </div>
        </div>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => handleOpenChange(false)}
            disabled={loading}
          >
            Cancel
          </Button>
          <Button
            className="bg-violet-600"
            onClick={handleSave}
            disabled={loading || !filename.trim()}
          >
            {loading ? "Saving..." : "Save Changes"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
