import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import type { Document } from "@/lib/api";

type DeleteDocumentDialogProps = {
  document: Document | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDelete: (documentId: string) => Promise<void>;
};

export function DeleteDocumentDialog({
  document,
  open,
  onOpenChange,
  onDelete,
}: DeleteDocumentDialogProps) {
  const [loading, setLoading] = useState(false);

  const handleDelete = async () => {
    if (!document) return;

    setLoading(true);
    try {
      await onDelete(document._id);
      onOpenChange(false);
    } catch (error) {
      console.error("Failed to delete document:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenChange = (newOpen: boolean) => {
    if (!loading) {
      onOpenChange(newOpen);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Delete Document</DialogTitle>
          <DialogDescription>
            Are you sure you want to delete "{document?.filename}"? This action
            cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => handleOpenChange(false)}
            disabled={loading}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={loading}
            className="bg-violet-600"
          >
            {loading ? "Deleting..." : "Delete"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}